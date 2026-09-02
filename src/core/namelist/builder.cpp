#include "core/namelist/builder.hpp"

#if TENRYU_ENABLE_PYTHON

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pybind11/eval.h>
#include <pybind11/stl.h>

#include "core/auto_zone.hpp"
#include "core/cone_shell_ladder.hpp"
#include "core/config_validate.hpp"
#include "core/error.hpp"
#include "core/radiation_group_structure.hpp"
#include "hydro/pressure_drive_perturbation.cuh"
#include "core/zoning_intent.hpp"
#include "materials/eos_table.hpp"
#include "materials/ionmix_reader.hpp"
#include "materials/tmat_reader.hpp"

namespace tenryu::core::namelist {
namespace {

namespace py = pybind11;

thread_local Builder* g_active_builder = nullptr;

constexpr std::uint64_t kFnv1aOffsetBasis = 1469598103934665603ull;
constexpr std::uint64_t kFnv1aPrime = 1099511628211ull;

void fnv1a_mix_bytes(std::uint64_t& hash, const void* data, const std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= static_cast<std::uint64_t>(bytes[i]);
    hash *= kFnv1aPrime;
  }
}

template <typename T>
void fnv1a_mix_value(std::uint64_t& hash, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>, "FNV mixer requires POD value");
  fnv1a_mix_bytes(hash, &value, sizeof(T));
}

void fnv1a_mix_string(std::uint64_t& hash, const std::string& value) {
  const std::uint64_t size = static_cast<std::uint64_t>(value.size());
  fnv1a_mix_value(hash, size);
  if (!value.empty()) {
    fnv1a_mix_bytes(hash, value.data(), value.size());
  }
}

std::uint64_t compute_material_eos_signature(const Config::MaterialsConfig::MatDef& mat) {
  if (mat.eos_model == "ideal_gas" || !mat.eos_tables) {
    return 0;
  }
  const auto& table = mat.eos_tables->total;
  if (table.rho_grid.empty() || table.T_grid_eV.empty()) {
    return 0;
  }

  std::uint64_t hash = kFnv1aOffsetBasis;
  fnv1a_mix_string(hash, mat.eos_model);
  fnv1a_mix_string(hash, mat.eos_file);
  fnv1a_mix_string(hash, mat.hydro_eos_backend);

  const std::uint64_t n_rho = static_cast<std::uint64_t>(table.n_rho());
  const std::uint64_t n_T = static_cast<std::uint64_t>(table.n_T());
  const double rho_min = table.rho_grid.front();
  const double rho_max = table.rho_grid.back();
  const double T_min_eV = table.T_grid_eV.front();
  const double T_max_eV = table.T_grid_eV.back();

  fnv1a_mix_value(hash, n_rho);
  fnv1a_mix_value(hash, n_T);
  fnv1a_mix_value(hash, rho_min);
  fnv1a_mix_value(hash, rho_max);
  fnv1a_mix_value(hash, T_min_eV);
  fnv1a_mix_value(hash, T_max_eV);
  return hash;
}

bool valid_temperature_range(const std::vector<double>& range) {
  return range.size() == 2U && std::isfinite(range[0]) && std::isfinite(range[1]) &&
         range[0] > 0.0 && range[1] > range[0];
}

static std::string langdon_profile_compatibility_error(
    const Config::LaserConfig& laser) {
  if (laser.beams.empty()) {
    return {};
  }

  const auto effective_profile = [&laser](const Config::LaserConfig::BeamDef& beam) {
    const std::string model =
        beam.profile_model.empty() ? laser.profile_model : beam.profile_model;
    const double w0_um =
        (beam.profile_w0_um > 0.0) ? beam.profile_w0_um : laser.profile_w0_um;
    const int m = (beam.profile_m > 0) ? beam.profile_m : laser.profile_m;
    return std::tuple<std::string, double, int>{model, w0_um, m};
  };

  const auto [model0, w0_um0, m0] = effective_profile(laser.beams.front());
  if (model0 != "gaussian" && model0 != "super_gaussian" &&
      model0 != "flat_top") {
    return "profile model \"" + model0 + "\" is not supported";
  }
  for (std::size_t i = 1; i < laser.beams.size(); ++i) {
    const auto [model, w0_um, m] = effective_profile(laser.beams[i]);
    if (model != "gaussian" && model != "super_gaussian" &&
        model != "flat_top") {
      return "beam " + std::to_string(i) + " profile model \"" + model +
             "\" is not supported";
    }
    if (model != model0 || w0_um != w0_um0 || m != m0) {
      return "beams do not share one effective (model, w0, m) profile";
    }
  }
  return {};
}

void expand_temperature_range(std::vector<double>& range,
                              const double lo,
                              const double hi) {
  if (!(std::isfinite(lo) && std::isfinite(hi) && lo > 0.0 && hi > lo)) {
    return;
  }
  if (range.empty()) {
    range = {lo, hi};
    return;
  }
  range[0] = std::min(range[0], lo);
  range[1] = std::max(range[1], hi);
}

std::vector<double> opacity_temperature_range(
    const materials::IonmixOpacityData& opacity) {
  if (opacity.temps_eV.size() < 2U) {
    return {};
  }
  return {opacity.temps_eV.front(), opacity.temps_eV.back()};
}

std::vector<double> repack_energy_range(
    const Config::RadiationConfig& radiation,
    const materials::IonmixOpacityData& opacity) {
  if (valid_temperature_range(radiation.compute_T_range_eV)) {
    return radiation.compute_T_range_eV;
  }
  if (opacity.bounds_eV.size() >= 2U &&
      opacity.bounds_eV.front() > 0.0 &&
      opacity.bounds_eV.back() > opacity.bounds_eV.front()) {
    return {opacity.bounds_eV.front(), opacity.bounds_eV.back()};
  }
  return opacity_temperature_range(opacity);
}

std::string py_type_name(const py::handle value) {
  if (value.is_none()) {
    return "NoneType";
  }
  return py::str(py::type::of(value).attr("__name__")).cast<std::string>();
}

bool py_callable(const py::handle value) {
  return PyCallable_Check(value.ptr()) != 0;
}

std::string format_type_error(std::string_view path,
                              std::string_view expected,
                              const py::handle value) {
  return std::string(path) + " must be " + std::string(expected) + ", got " +
         py_type_name(value);
}

std::string format_range_error(std::string_view path,
                               std::string_view expected,
                               const std::string& got) {
  return std::string(path) + " must be " + std::string(expected) + ", got " + got;
}

[[noreturn]] void throw_value_type_error(std::string_view path,
                                         std::string_view expected,
                                         const py::handle value) {
  throw ValueError(format_type_error(path, expected, value));
}

[[noreturn]] void throw_cast_error(std::string_view path,
                                   std::string_view expected,
                                   const py::handle value) {
  throw ValueError(format_type_error(path, expected, value));
}

bool strict_bool(const py::handle value, std::string_view path) {
  if (!py::isinstance<py::bool_>(value)) {
    throw_value_type_error(path, "bool", value);
  }
  return value.cast<bool>();
}

std::string strict_string(const py::handle value, std::string_view path) {
  if (!py::isinstance<py::str>(value)) {
    throw_value_type_error(path, "str", value);
  }
  return value.cast<std::string>();
}

std::string resolve_namelist_relative_path(const Config& config,
                                           const std::string& raw_path) {
  if (raw_path.empty()) {
    return raw_path;
  }

  std::filesystem::path path(raw_path);
  if (path.is_absolute()) {
    return path.lexically_normal().string();
  }

  const std::filesystem::path base_dir(config.meta.namelist_source_dir);
  if (base_dir.empty()) {
    return path.lexically_normal().string();
  }

  return std::filesystem::absolute(base_dir / path).lexically_normal().string();
}

std::int64_t strict_int64(const py::handle value, std::string_view path) {
  if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
    throw_value_type_error(path, "int", value);
  }
  try {
    return value.cast<std::int64_t>();
  } catch (const py::cast_error&) {
    throw_cast_error(path, "int64", value);
  }
}

int strict_int32(const py::handle value, std::string_view path) {
  const std::int64_t v = strict_int64(value, path);
  if (v < static_cast<std::int64_t>(std::numeric_limits<int>::min()) ||
      v > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    throw ConfigError(format_range_error(path,
                                         "32-bit int range",
                                         std::to_string(v)));
  }
  return static_cast<int>(v);
}

std::uint64_t strict_uint64(const py::handle value, std::string_view path) {
  if (py::isinstance<py::bool_>(value) || !py::isinstance<py::int_>(value)) {
    throw_value_type_error(path, "int", value);
  }
  try {
    return value.cast<std::uint64_t>();
  } catch (const py::cast_error&) {
    throw_cast_error(path, "uint64", value);
  }
}

double numeric_as_double(const py::handle value, std::string_view path) {
  if (py::isinstance<py::bool_>(value)) {
    throw_value_type_error(path, "float", value);
  }
  if (!py::isinstance<py::float_>(value) && !py::isinstance<py::int_>(value)) {
    throw_value_type_error(path, "float", value);
  }
  double result;
  try {
    result = py::cast<double>(value);
  } catch (const py::cast_error&) {
    throw_cast_error(path, "float", value);
  }
  if (!std::isfinite(result)) {
    throw ValueError(format_range_error(path, "finite float", std::to_string(result)));
  }
  return result;
}

std::vector<double> strict_double_vector(const py::handle value, std::string_view path) {
  if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value)) {
    throw_value_type_error(path, "list[float]", value);
  }
  std::vector<double> out;
  const auto seq = py::reinterpret_borrow<py::sequence>(value);
  out.reserve(seq.size());
  for (std::size_t i = 0; i < seq.size(); ++i) {
    out.push_back(numeric_as_double(seq[i], std::string(path) + "[" + std::to_string(i) + "]"));
  }
  return out;
}

std::vector<int> strict_int_vector(const py::handle value, std::string_view path) {
  if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value)) {
    throw_value_type_error(path, "list[int]", value);
  }
  std::vector<int> out;
  const auto seq = py::reinterpret_borrow<py::sequence>(value);
  out.reserve(seq.size());
  for (std::size_t i = 0; i < seq.size(); ++i) {
    out.push_back(strict_int32(seq[i], std::string(path) + "[" + std::to_string(i) + "]"));
  }
  return out;
}

std::vector<std::string> strict_string_vector(const py::handle value,
                                              std::string_view path) {
  if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value)) {
    throw_value_type_error(path, "list[str]", value);
  }
  std::vector<std::string> out;
  const auto seq = py::reinterpret_borrow<py::sequence>(value);
  out.reserve(seq.size());
  for (std::size_t i = 0; i < seq.size(); ++i) {
    out.push_back(strict_string(
        seq[i], std::string(path) + "[" + std::to_string(i) + "]"));
  }
  return out;
}

using PressureDrivePerturbationConfig =
    Config::NumericsConfig::HydroConfig::PressureDrivePerturbationConfig;

double pressure_drive_amplitude_as_double(const py::handle value,
                                          std::string_view path) {
  if (py::isinstance<py::bool_>(value) ||
      (!py::isinstance<py::float_>(value) &&
       !py::isinstance<py::int_>(value))) {
    throw_value_type_error(path, "float", value);
  }
  double result;
  try {
    result = py::cast<double>(value);
  } catch (const py::cast_error&) {
    throw_cast_error(path, "float", value);
  }
  if (!std::isfinite(result)) {
    throw ConfigError(format_range_error(
        path, "finite float", std::to_string(result)));
  }
  return result;
}

void resolve_random_pressure_drive_modes(
    PressureDrivePerturbationConfig& config) {
  config.random_enabled = config.random_rms > 0.0;
  if (!config.random_enabled) {
    return;
  }

  const int random_count = config.random_l_max - config.random_l_min + 1;
  if (config.mode_l.size() + static_cast<std::size_t>(random_count) >
      static_cast<std::size_t>(
          tenryu::hydro::PressureDrivePerturbationParams::kMaxModes)) {
    throw ConfigError(
        "Numerics.hydro.pressure_drive_perturbation has more than 24 "
        "resolved modes");
  }

  std::vector<double> normal_draws;
  normal_draws.reserve(static_cast<std::size_t>(random_count));
  double weighted_square_sum = 0.0;
  for (int l = config.random_l_min; l <= config.random_l_max; ++l) {
    std::uint64_t x =
        static_cast<std::uint64_t>(config.random_seed) ^
        (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(l + 1));
    const auto next = [&x]() {
      x += 0x9E3779B97F4A7C15ULL;
      std::uint64_t z = x;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      return z ^ (z >> 31);
    };
    double u1 = static_cast<double>(next() >> 11) * 0x1.0p-53;
    const double u2 = static_cast<double>(next() >> 11) * 0x1.0p-53;
    if (u1 <= 0.0) {
      u1 = 0x1.0p-53;
    }
    const double n_l =
        std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    normal_draws.push_back(n_l);
    weighted_square_sum += n_l * n_l / static_cast<double>(2 * l + 1);
  }
  if (weighted_square_sum == 0.0) {
    throw ConfigError(
        "Numerics.hydro.pressure_drive_perturbation random spectrum has "
        "zero normalization");
  }

  const double scale = config.random_rms / std::sqrt(weighted_square_sum);
  for (int l = config.random_l_min; l <= config.random_l_max; ++l) {
    config.mode_l.push_back(l);
    config.mode_a.push_back(
        scale * normal_draws[static_cast<std::size_t>(l -
                                                     config.random_l_min)]);
  }

  std::ostringstream resolved;
  resolved << "Numerics.hydro.pressure_drive_perturbation resolved modes:";
  resolved << std::setprecision(17);
  for (std::size_t k = 0; k < config.mode_l.size(); ++k) {
    resolved << " (" << config.mode_l[k] << "," << config.mode_a[k] << ")";
  }
  tenryu::core::log_info(resolved.str());
}

void compute_pressure_drive_perturbation_range(
    PressureDrivePerturbationConfig& config) {
  const tenryu::hydro::PressureDrivePerturbationParams params =
      make_pressure_drive_perturbation_params(config);
  constexpr int kIntervals = 4096;
  double g_min = std::numeric_limits<double>::infinity();
  double g_max = -std::numeric_limits<double>::infinity();
  double theta_at_min = 0.0;
  for (int i = 0; i <= kIntervals; ++i) {
    const double theta =
        static_cast<double>(i) * M_PI / static_cast<double>(kIntervals);
    const double g = tenryu::hydro::pressure_drive_perturbation_g(
        params, std::sin(theta), std::cos(theta));
    if (g < g_min) {
      g_min = g;
      theta_at_min = theta;
    }
    g_max = std::max(g_max, g);
  }
  if (!(std::isfinite(g_min) && std::isfinite(g_max))) {
    throw ConfigError(
        "Numerics.hydro.pressure_drive_perturbation range is not finite");
  }
  config.g_min = g_min;
  config.g_max = g_max;
  if (g_min <= 0.0) {
    std::ostringstream message;
    message << std::setprecision(17)
            << "Numerics.hydro.pressure_drive_perturbation profile is "
               "non-positive at theta="
            << theta_at_min << ": g=" << g_min;
    throw ConfigError(message.str());
  }
}

std::string normalize_repr(std::string repr) {
  static const std::regex kAddressRegex{"0x[0-9a-fA-F]+"};
  return std::regex_replace(std::move(repr), kAddressRegex, "<addr>");
}

std::string sha256_string(const std::string& text) {
  try {
    py::object digest = py::module_::import("hashlib").attr("sha256")(py::bytes(text));
    return "sha256:" + py::str(digest.attr("hexdigest")()).cast<std::string>();
  } catch (...) {
    return "unavailable";
  }
}

PythonCallable extract_callable_or_throw(const py::handle value, std::string_view path) {
  if (py::isinstance<py::function>(value)) {
    const auto f = py::reinterpret_borrow<py::function>(value);
    PythonCallable info;
    info.detected = true;
    try {
      info.name = py::str(value.attr("__name__")).cast<std::string>();
    } catch (...) {
      info.name = "<anonymous>";
    }
    info.repr = normalize_repr(py::repr(value).cast<std::string>());

    try {
      py::object source_obj = py::module_::import("inspect").attr("getsource")(value);
      info.source_hash = sha256_string(py::str(source_obj).cast<std::string>());
    } catch (...) {
      info.source_hash = "unavailable";
    }

    py::object test_result;
    try {
      auto test_val = f(0.0);
      test_result = py::reinterpret_borrow<py::object>(test_val);
    } catch (const py::error_already_set&) {
      try {
        auto test_val = f(0.0, 0.0);
        test_result = py::reinterpret_borrow<py::object>(test_val);
      } catch (const py::error_already_set& probe_err) {
        throw ConfigError(std::string(path) +
                          " callable probe failed at f(0.0) and f(0.0,0.0): " +
                          probe_err.what());
      }
    }

    const bool is_velocity = path == "Geometry.velocity";
    if (is_velocity && py::isinstance<py::sequence>(test_result) &&
        !py::isinstance<py::str>(test_result)) {
      const py::sequence seq = py::reinterpret_borrow<py::sequence>(test_result);
      if (seq.size() != 2) {
        throw ConfigError(std::string(path) +
                          " callable probe must return finite float or 2-component sequence");
      }
      for (std::size_t i = 0; i < seq.size(); ++i) {
        double component = 0.0;
        try {
          component = py::cast<double>(seq[i]);
        } catch (const py::cast_error&) {
          throw ConfigError(std::string(path) +
                            " callable probe must return finite float or 2-component sequence");
        }
        if (!std::isfinite(component)) {
          throw ConfigError(std::string(path) +
                            " callable probe must return finite float or finite 2-component sequence");
        }
      }
    } else {
      double test_val = 0.0;
      try {
        test_val = py::cast<double>(test_result);
      } catch (const py::cast_error&) {
        throw ConfigError(std::string(path) +
                          " callable probe must return a finite float");
      }
      if (!std::isfinite(test_val)) {
        throw ConfigError(std::string(path) +
                          " callable probe must return a finite float");
      }
    }

    return info;
  }

  if (py_callable(value)) {
    throw ConfigError(std::string(path) +
                      " must be a function (def/lambda), got " + py_type_name(value));
  }

  throw ValueError(format_type_error(path, "callable", value));
}

Config::CallableInfo to_config_callable(const PythonCallable& callable) {
  Config::CallableInfo out;
  out.name = callable.name;
  out.repr = callable.repr;
  out.source_hash = callable.source_hash;
  out.detected = callable.detected;
  return out;
}

bool allow_unknown_keys() {
  const char* allow = std::getenv("TENRYU_NAMELIST_ALLOW_UNKNOWN_KEYS");
  return allow != nullptr && std::string_view{allow} == "1";
}

int levenshtein_distance(std::string_view a, std::string_view b) {
  if (a.empty()) {
    return static_cast<int>(b.size());
  }
  if (b.empty()) {
    return static_cast<int>(a.size());
  }

  std::vector<int> prev(b.size() + 1);
  std::vector<int> curr(b.size() + 1);
  for (std::size_t j = 0; j <= b.size(); ++j) {
    prev[j] = static_cast<int>(j);
  }
  for (std::size_t i = 1; i <= a.size(); ++i) {
    curr[0] = static_cast<int>(i);
    for (std::size_t j = 1; j <= b.size(); ++j) {
      const int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
      curr[j] = std::min({prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost});
    }
    std::swap(prev, curr);
  }
  return prev[b.size()];
}

std::string suggest_key(std::string_view unknown,
                        const std::initializer_list<std::string_view>& allowed) {
  int best_distance = std::numeric_limits<int>::max();
  std::string best;
  for (const std::string_view candidate : allowed) {
    const int distance = levenshtein_distance(unknown, candidate);
    if (distance < best_distance) {
      best_distance = distance;
      best = std::string(candidate);
    }
  }
  if (best_distance <= 3) {
    return best;
  }
  return {};
}

void enforce_known_keys(const py::dict& kwargs,
                        std::string_view path,
                        const std::initializer_list<std::string_view>& allowed) {
  std::unordered_set<std::string> allowed_keys;
  allowed_keys.reserve(allowed.size());
  for (const std::string_view key : allowed) {
    allowed_keys.insert(std::string(key));
  }

  const bool allow_unknown = allow_unknown_keys();
  for (const auto item : kwargs) {
    const std::string key = py::str(item.first).cast<std::string>();
    if (allowed_keys.contains(key)) {
      continue;
    }

    const std::string suggestion = suggest_key(key, allowed);
    if (allow_unknown) {
      if (suggestion.empty()) {
        tenryu::core::log_warning(std::string(path) + ": unknown key '" + key +
                                  "' ignored");
      } else {
        tenryu::core::log_warning(std::string(path) + ": unknown key '" + key +
                                  "' ignored (did you mean '" + suggestion + "'?)");
      }
      continue;
    }

    std::ostringstream oss;
    oss << path << "." << key << " is not a supported key";
    if (!suggestion.empty()) {
      oss << " (did you mean '" << suggestion << "'?)";
    }
    throw ConfigError(oss.str());
  }
}

bool has_key(const py::dict& kwargs, const char* key) {
  return kwargs.contains(py::str(key));
}

void parse_euler_window_dict(
    const py::dict& euler_window,
    const std::string& path,
    Config::NumericsConfig::AleConfig::EulerWindowConfig& config) {
  enforce_known_keys(
      euler_window,
      path,
      {"enabled", "role", "shape", "r0", "r1", "z0", "z1", "cr", "cz",
       "rad_in", "rad_out", "transition_width", "t_on_s", "t_off_s",
       "feather_min_layers", "guard_layers", "axis_core_transaction_mode",
       "replay_table_path", "replay_tau_lead", "replay_tau_splice",
       "replay_beta",
       "axis_core_transition_passage_enabled",
       "axis_core_ring_release_enabled"});
  if (has_key(euler_window, "enabled")) {
    config.enabled =
        strict_bool(euler_window["enabled"], path + ".enabled");
  }
  if (has_key(euler_window, "shape")) {
    config.shape =
        strict_string(euler_window["shape"], path + ".shape");
    if (config.shape != "rectangle" && config.shape != "annulus" &&
        config.shape != "spherical_disk") {
      throw ValueError(
          path + ".shape must be one of {\"rectangle\", \"annulus\", "
                 "\"spherical_disk\"}");
    }
  }
  if (has_key(euler_window, "role")) {
    config.role = strict_string(euler_window["role"], path + ".role");
    if (config.role != "" && config.role != "axis_survival_core") {
      throw ConfigError(
          path + ".role must be one of {\"\", \"axis_survival_core\"}");
    }
  }
  if (has_key(euler_window, "r0")) {
    config.r0 = numeric_as_double(euler_window["r0"], path + ".r0");
  }
  if (has_key(euler_window, "r1")) {
    config.r1 = numeric_as_double(euler_window["r1"], path + ".r1");
  }
  if (has_key(euler_window, "z0")) {
    config.z0 = numeric_as_double(euler_window["z0"], path + ".z0");
  }
  if (has_key(euler_window, "z1")) {
    config.z1 = numeric_as_double(euler_window["z1"], path + ".z1");
  }
  if (has_key(euler_window, "cr")) {
    config.cr = numeric_as_double(euler_window["cr"], path + ".cr");
  }
  if (has_key(euler_window, "cz")) {
    config.cz = numeric_as_double(euler_window["cz"], path + ".cz");
  }
  if (has_key(euler_window, "rad_in")) {
    config.rad_in =
        numeric_as_double(euler_window["rad_in"], path + ".rad_in");
  }
  if (has_key(euler_window, "rad_out")) {
    config.rad_out =
        numeric_as_double(euler_window["rad_out"], path + ".rad_out");
  }
  if (has_key(euler_window, "transition_width")) {
    config.transition_width = numeric_as_double(
        euler_window["transition_width"], path + ".transition_width");
  }
  if (has_key(euler_window, "t_on_s")) {
    config.t_on_s =
        numeric_as_double(euler_window["t_on_s"], path + ".t_on_s");
  }
  if (has_key(euler_window, "t_off_s")) {
    config.t_off_s =
        numeric_as_double(euler_window["t_off_s"], path + ".t_off_s");
  }
  if (has_key(euler_window, "feather_min_layers")) {
    config.feather_min_layers = strict_int32(
        euler_window["feather_min_layers"], path + ".feather_min_layers");
  }
  if (has_key(euler_window, "guard_layers")) {
    config.guard_layers = strict_int32(
        euler_window["guard_layers"], path + ".guard_layers");
  }
  if (has_key(euler_window, "axis_core_transaction_mode")) {
    config.axis_core_transaction_mode = strict_string(
        euler_window["axis_core_transaction_mode"],
        path + ".axis_core_transaction_mode");
  }
  if (has_key(euler_window, "replay_table_path")) {
    config.replay_table_path = strict_string(
        euler_window["replay_table_path"], path + ".replay_table_path");
  }
  if (has_key(euler_window, "replay_tau_lead")) {
    config.replay_tau_lead = numeric_as_double(
        euler_window["replay_tau_lead"], path + ".replay_tau_lead");
  }
  if (has_key(euler_window, "replay_tau_splice")) {
    config.replay_tau_splice = numeric_as_double(
        euler_window["replay_tau_splice"], path + ".replay_tau_splice");
  }
  if (has_key(euler_window, "replay_beta")) {
    config.replay_beta = numeric_as_double(
        euler_window["replay_beta"], path + ".replay_beta");
  }
  if (has_key(euler_window, "axis_core_transition_passage_enabled")) {
    config.axis_core_transition_passage_enabled = strict_bool(
        euler_window["axis_core_transition_passage_enabled"],
        path + ".axis_core_transition_passage_enabled");
  }
  if (has_key(euler_window, "axis_core_ring_release_enabled")) {
    config.axis_core_ring_release_enabled = strict_bool(
        euler_window["axis_core_ring_release_enabled"],
        path + ".axis_core_ring_release_enabled");
  }
}

void validate_euler_window_config(
    const Config::NumericsConfig::AleConfig::EulerWindowConfig& config,
    const std::string& path) {
  if (config.axis_core_transaction_mode != "static" &&
      config.axis_core_transaction_mode != "always_moving" &&
      config.axis_core_transaction_mode != "clearance_replay") {
    throw ValueError(
        path + ".axis_core_transaction_mode must be one of "
               "{\"static\", \"always_moving\", \"clearance_replay\"}");
  }
  if (config.axis_core_transaction_mode != "static" &&
      (!config.enabled || config.role != "axis_survival_core")) {
    throw ConfigError(
        path + " axis_core_transaction_mode=\"" +
        config.axis_core_transaction_mode + "\" requires "
               "enabled=true and role=\"axis_survival_core\"");
  }
  if (!(std::isfinite(config.replay_tau_lead) &&
        config.replay_tau_lead > 0.0)) {
    throw ValueError(path + ".replay_tau_lead must be finite and > 0");
  }
  if (!(std::isfinite(config.replay_tau_splice) &&
        config.replay_tau_splice > 0.0)) {
    throw ValueError(path + ".replay_tau_splice must be finite and > 0");
  }
  if (!(std::isfinite(config.replay_beta) && config.replay_beta >= 0.0)) {
    throw ValueError(path + ".replay_beta must be finite and >= 0");
  }
  if (config.axis_core_transition_passage_enabled &&
      (!config.enabled || config.role != "axis_survival_core")) {
    throw ConfigError(
        path + " axis_core_transition_passage_enabled=true requires "
               "enabled=true and role=\"axis_survival_core\"");
  }
  if (config.axis_core_ring_release_enabled &&
      (!config.enabled || config.role != "axis_survival_core")) {
    throw ConfigError(
        path + " axis_core_ring_release_enabled=true requires "
               "enabled=true and role=\"axis_survival_core\"");
  }
  if (config.role != "" && config.role != "axis_survival_core") {
    throw ConfigError(
        path + ".role must be one of {\"\", \"axis_survival_core\"}");
  }
  if (config.shape != "rectangle" && config.shape != "annulus" &&
      config.shape != "spherical_disk") {
    throw ValueError(
        path + ".shape must be one of {\"rectangle\", \"annulus\", "
               "\"spherical_disk\"}");
  }
  if (config.shape == "spherical_disk" &&
      config.role != "axis_survival_core") {
    throw ConfigError(
        path + " shape=\"spherical_disk\" requires "
               "role=\"axis_survival_core\"");
  }
  if (config.role == "axis_survival_core") {
    if (config.shape != "spherical_disk") {
      throw ConfigError(
          path + " role=\"axis_survival_core\" requires "
                 "shape=\"spherical_disk\"");
    }
    if (config.t_on_s != 0.0) {
      throw ConfigError(
          path + " role=\"axis_survival_core\" requires t_on_s == 0.0");
    }
    if (config.t_off_s >= 0.0) {
      throw ConfigError(
          "the axis-core time-gated release was deleted 2026-08-03; "
          "t_off_s must be -1 for axis_survival_core");
    }
    if (config.rad_in != 0.0) {
      throw ConfigError(
          path + " shape=\"spherical_disk\" requires rad_in == 0.0");
    }
    if (config.cr != 0.0 || config.cz != 0.0) {
      throw ConfigError(
          path + " shape=\"spherical_disk\" requires cr == 0.0 and "
                 "cz == 0.0");
    }
  }
  if (!config.enabled) {
    return;
  }
  if (!(config.transition_width > 0.0)) {
    throw ValueError(
        path + ".transition_width must be > 0 when enabled");
  }
  if (config.t_off_s >= 0.0 && !(config.t_off_s > config.t_on_s)) {
    throw ValueError(path + ".t_off_s must exceed t_on_s");
  }
  if (config.shape == "rectangle" &&
      (!(config.r1 > config.r0) || !(config.z1 > config.z0))) {
    throw ValueError(
        path + " rectangle bounds require r1 > r0 and z1 > z0");
  }
  if (config.shape == "annulus" &&
      (!(config.rad_in >= 0.0) || !(config.rad_out > config.rad_in))) {
    throw ValueError(
        path + " annulus bounds require rad_out > rad_in >= 0");
  }
}

void parse_band_ale_dict(
    const py::dict& band_ale,
    const std::string& path,
    Config::NumericsConfig::AleConfig::BandAleConfig& config) {
  enforce_known_keys(
      band_ale,
      path,
      {"enabled",
       "aspect_trigger",
       "release_hysteresis",
       "chi",
       "respace_move_cap_frac",
       "estimator_band_hold_mach",
       "bands",
       "compose_with_rezone",
       "belt_target",
       "center_target",
       "axis_target",
       "axis_segment_halfwidth",
       "axis_shell_block_enabled",
       "sigma_linesearch_enabled",
       "transaction_energy_closure_enabled",
       "estimator_band_cut",
       "estimator_band_shock_hold",
       "estimator_band_front_hold_margin_rows",
       "estimator_band_axis",
       "estimator_band_in_rows",
       "estimator_band_out_rows",
       "estimator_band_eta_on",
       "estimator_band_eta_off",
       "estimator_band_per_column",
       "estimator_band_pc_filter_halfwidth",
       "estimator_band_pc_slope_limit",
       "estimator_band_pc_slope_reject",
       "estimator_band_pc_curvature_limit",
       "estimator_band_pc_chi_max",
       "estimator_band_pc_chi_step",
       "estimator_band_pc_sigma_floor",
       "estimator_band_pc_coverage_full",
       "estimator_band_pc_coverage_min",
       "estimator_band_pc_cooldown_events",
       "estimator_band_pc_phase_b",
       "estimator_band_pc_tube_dilate_rows",
       "estimator_band_pc_tube_dilate_cols_extra",
       "estimator_band_pc_ambiguous_hold_fraction",
       "closure_catchment_enabled",
       "closure_catchment_forced_active",
       "closure_catchment_s_catch_cm",
       "closure_catchment_s_protect_cm",
       "closure_catchment_spacing_floor_cm",
       "closure_catchment_ratio_max",
       "closure_catchment_nu_max",
       "closure_catchment_max_bites",
       "closure_catchment_shock_hold",
       "closure_catchment_eta_h_arm",
       "closure_catchment_eta_h_full",
       "closure_catchment_eta_m_arm",
       "closure_catchment_eta_m_full",
       "closure_catchment_reset_eta",
       "closure_catchment_support_core_rows",
       "closure_catchment_support_taper_rows",
       "closure_catchment_accum_frac",
       "closure_catchment_rearm_drop",
       "pole_theta_enabled",
       "pole_theta_routine_enabled",
       "pole_theta_phys_lp",
       "pole_theta_phys_lc",
       "pole_theta_noise_floor",
       "pole_theta_noise_ceiling",
       "pole_theta_h_arm",
       "pole_theta_h_fire",
       "pole_theta_h_hard",
       "pole_theta_h_release",
       "pole_theta_kappa_arm",
       "pole_theta_kappa_fire",
       "pole_theta_kappa_hard",
       "pole_theta_alpha",
       "pole_theta_alpha_hard",
       "pole_theta_deadband_frac",
       "pole_theta_move_limit_frac",
       "pole_theta_move_limit_hard_frac",
       "pole_theta_cooldown_s",
       "pole_theta_cooldown_base_s",
       "pole_theta_predict_window_s",
       "pole_theta_predict_horizon_s",
       "pole_theta_halo_columns",
       "pole_theta_halo_rows",
       "pole_theta_post_h_floor",
       "pole_theta_curve_preserving",
       "pole_theta_protected_modes",
       "pole_theta_fit_order",
       "pole_theta_shock_hold",
       "shell_window_in_rows",
       "shell_window_out_rows",
       "shell_boundary_guard_rows",
       "shell_min_spacing_frac",
       "shell_front_metric",
       "shell_target",
       "axis_repair_enabled",
       "axis_repair_eta_on",
       "axis_repair_eta_off",
       "axis_repair_cap_rel"});
  if (has_key(band_ale, "enabled")) {
    config.enabled = strict_bool(band_ale["enabled"], path + ".enabled");
  }
  if (has_key(band_ale, "aspect_trigger")) {
    config.aspect_trigger =
        numeric_as_double(band_ale["aspect_trigger"], path + ".aspect_trigger");
  }
  if (has_key(band_ale, "release_hysteresis")) {
    config.release_hysteresis = numeric_as_double(
        band_ale["release_hysteresis"], path + ".release_hysteresis");
  }
  if (has_key(band_ale, "chi")) {
    config.chi = numeric_as_double(band_ale["chi"], path + ".chi");
  }
  if (has_key(band_ale, "respace_move_cap_frac")) {
    config.respace_move_cap_frac = numeric_as_double(
        band_ale["respace_move_cap_frac"],
        path + ".respace_move_cap_frac");
  }
  if (has_key(band_ale, "estimator_band_hold_mach")) {
    config.estimator_band_hold_mach = numeric_as_double(
        band_ale["estimator_band_hold_mach"],
        path + ".estimator_band_hold_mach");
  }
  if (has_key(band_ale, "bands")) {
    config.bands = strict_string(band_ale["bands"], path + ".bands");
  }
  if (has_key(band_ale, "compose_with_rezone")) {
    config.compose_with_rezone = strict_bool(
        band_ale["compose_with_rezone"], path + ".compose_with_rezone");
  }
  if (has_key(band_ale, "belt_target")) {
    config.belt_target =
        strict_string(band_ale["belt_target"], path + ".belt_target");
  }
  if (has_key(band_ale, "center_target")) {
    config.center_target =
        strict_string(band_ale["center_target"], path + ".center_target");
  }
  if (has_key(band_ale, "axis_target")) {
    config.axis_target =
        strict_string(band_ale["axis_target"], path + ".axis_target");
  }
  if (has_key(band_ale, "axis_segment_halfwidth")) {
    config.axis_segment_halfwidth = strict_int32(
        band_ale["axis_segment_halfwidth"],
        path + ".axis_segment_halfwidth");
  }
  if (has_key(band_ale, "axis_shell_block_enabled")) {
    config.axis_shell_block_enabled = strict_bool(
        band_ale["axis_shell_block_enabled"],
        path + ".axis_shell_block_enabled");
  }
  if (has_key(band_ale, "sigma_linesearch_enabled")) {
    config.sigma_linesearch_enabled = strict_bool(
        band_ale["sigma_linesearch_enabled"],
        path + ".sigma_linesearch_enabled");
  }
  if (has_key(band_ale, "transaction_energy_closure_enabled")) {
    config.transaction_energy_closure_enabled = strict_bool(
        band_ale["transaction_energy_closure_enabled"],
        path + ".transaction_energy_closure_enabled");
  }
  if (has_key(band_ale, "estimator_band_cut")) {
    config.estimator_band_cut = numeric_as_double(
        band_ale["estimator_band_cut"], path + ".estimator_band_cut");
  }
  if (has_key(band_ale, "estimator_band_shock_hold")) {
    config.estimator_band_shock_hold = numeric_as_double(
        band_ale["estimator_band_shock_hold"],
        path + ".estimator_band_shock_hold");
  }
  if (has_key(band_ale, "estimator_band_front_hold_margin_rows")) {
    config.estimator_band_front_hold_margin_rows = numeric_as_double(
        band_ale["estimator_band_front_hold_margin_rows"],
        path + ".estimator_band_front_hold_margin_rows");
  }
  if (has_key(band_ale, "estimator_band_axis")) {
    config.estimator_band_axis = strict_string(
        band_ale["estimator_band_axis"], path + ".estimator_band_axis");
  }
  if (has_key(band_ale, "estimator_band_in_rows")) {
    config.estimator_band_in_rows = strict_int32(
        band_ale["estimator_band_in_rows"],
        path + ".estimator_band_in_rows");
  }
  if (has_key(band_ale, "estimator_band_out_rows")) {
    config.estimator_band_out_rows = strict_int32(
        band_ale["estimator_band_out_rows"],
        path + ".estimator_band_out_rows");
  }
  if (has_key(band_ale, "estimator_band_eta_on")) {
    config.estimator_band_eta_on = numeric_as_double(
        band_ale["estimator_band_eta_on"],
        path + ".estimator_band_eta_on");
  }
  if (has_key(band_ale, "estimator_band_eta_off")) {
    config.estimator_band_eta_off = numeric_as_double(
        band_ale["estimator_band_eta_off"],
        path + ".estimator_band_eta_off");
  }
  if (has_key(band_ale, "estimator_band_per_column")) {
    config.estimator_band_per_column = strict_bool(
        band_ale["estimator_band_per_column"],
        path + ".estimator_band_per_column");
  }
  if (has_key(band_ale, "estimator_band_pc_filter_halfwidth")) {
    config.estimator_band_pc_filter_halfwidth = strict_int32(
        band_ale["estimator_band_pc_filter_halfwidth"],
        path + ".estimator_band_pc_filter_halfwidth");
  }
  if (has_key(band_ale, "estimator_band_pc_slope_limit")) {
    config.estimator_band_pc_slope_limit = numeric_as_double(
        band_ale["estimator_band_pc_slope_limit"],
        path + ".estimator_band_pc_slope_limit");
  }
  if (has_key(band_ale, "estimator_band_pc_slope_reject")) {
    config.estimator_band_pc_slope_reject = numeric_as_double(
        band_ale["estimator_band_pc_slope_reject"],
        path + ".estimator_band_pc_slope_reject");
  }
  if (has_key(band_ale, "estimator_band_pc_curvature_limit")) {
    config.estimator_band_pc_curvature_limit = numeric_as_double(
        band_ale["estimator_band_pc_curvature_limit"],
        path + ".estimator_band_pc_curvature_limit");
  }
  if (has_key(band_ale, "estimator_band_pc_chi_max")) {
    config.estimator_band_pc_chi_max = numeric_as_double(
        band_ale["estimator_band_pc_chi_max"],
        path + ".estimator_band_pc_chi_max");
  }
  if (has_key(band_ale, "estimator_band_pc_chi_step")) {
    config.estimator_band_pc_chi_step = numeric_as_double(
        band_ale["estimator_band_pc_chi_step"],
        path + ".estimator_band_pc_chi_step");
  }
  if (has_key(band_ale, "estimator_band_pc_sigma_floor")) {
    config.estimator_band_pc_sigma_floor = numeric_as_double(
        band_ale["estimator_band_pc_sigma_floor"],
        path + ".estimator_band_pc_sigma_floor");
  }
  if (has_key(band_ale, "estimator_band_pc_coverage_full")) {
    config.estimator_band_pc_coverage_full = numeric_as_double(
        band_ale["estimator_band_pc_coverage_full"],
        path + ".estimator_band_pc_coverage_full");
  }
  if (has_key(band_ale, "estimator_band_pc_coverage_min")) {
    config.estimator_band_pc_coverage_min = numeric_as_double(
        band_ale["estimator_band_pc_coverage_min"],
        path + ".estimator_band_pc_coverage_min");
  }
  if (has_key(band_ale, "estimator_band_pc_cooldown_events")) {
    config.estimator_band_pc_cooldown_events = strict_int32(
        band_ale["estimator_band_pc_cooldown_events"],
        path + ".estimator_band_pc_cooldown_events");
  }
  if (has_key(band_ale, "estimator_band_pc_phase_b")) {
    config.estimator_band_pc_phase_b = strict_bool(
        band_ale["estimator_band_pc_phase_b"],
        path + ".estimator_band_pc_phase_b");
  }
  if (has_key(band_ale, "estimator_band_pc_tube_dilate_rows")) {
    config.estimator_band_pc_tube_dilate_rows = strict_int32(
        band_ale["estimator_band_pc_tube_dilate_rows"], path + ".estimator_band_pc_tube_dilate_rows");
  }
  if (has_key(band_ale, "estimator_band_pc_tube_dilate_cols_extra")) {
    config.estimator_band_pc_tube_dilate_cols_extra = strict_int32(
        band_ale["estimator_band_pc_tube_dilate_cols_extra"],
        path + ".estimator_band_pc_tube_dilate_cols_extra");
  }
  if (has_key(band_ale, "estimator_band_pc_ambiguous_hold_fraction")) {
    config.estimator_band_pc_ambiguous_hold_fraction = numeric_as_double(
        band_ale["estimator_band_pc_ambiguous_hold_fraction"],
        path + ".estimator_band_pc_ambiguous_hold_fraction");
  }
  if (has_key(band_ale, "closure_catchment_enabled")) {
    config.closure_catchment_enabled = strict_bool(
        band_ale["closure_catchment_enabled"],
        path + ".closure_catchment_enabled");
  }
  if (has_key(band_ale, "closure_catchment_forced_active")) {
    config.closure_catchment_forced_active = strict_bool(
        band_ale["closure_catchment_forced_active"],
        path + ".closure_catchment_forced_active");
  }
  if (has_key(band_ale, "closure_catchment_s_catch_cm")) {
    config.closure_catchment_s_catch_cm = numeric_as_double(
        band_ale["closure_catchment_s_catch_cm"],
        path + ".closure_catchment_s_catch_cm");
  }
  if (has_key(band_ale, "closure_catchment_s_protect_cm")) {
    config.closure_catchment_s_protect_cm = numeric_as_double(
        band_ale["closure_catchment_s_protect_cm"],
        path + ".closure_catchment_s_protect_cm");
  }
  if (has_key(band_ale, "closure_catchment_spacing_floor_cm")) {
    config.closure_catchment_spacing_floor_cm = numeric_as_double(
        band_ale["closure_catchment_spacing_floor_cm"],
        path + ".closure_catchment_spacing_floor_cm");
  }
  if (has_key(band_ale, "closure_catchment_ratio_max")) {
    config.closure_catchment_ratio_max = numeric_as_double(
        band_ale["closure_catchment_ratio_max"],
        path + ".closure_catchment_ratio_max");
  }
  if (has_key(band_ale, "closure_catchment_nu_max")) {
    config.closure_catchment_nu_max = numeric_as_double(
        band_ale["closure_catchment_nu_max"],
        path + ".closure_catchment_nu_max");
  }
  if (has_key(band_ale, "closure_catchment_max_bites")) {
    config.closure_catchment_max_bites = strict_int32(
        band_ale["closure_catchment_max_bites"],
        path + ".closure_catchment_max_bites");
  }
  if (has_key(band_ale, "closure_catchment_shock_hold")) {
    config.closure_catchment_shock_hold = numeric_as_double(
        band_ale["closure_catchment_shock_hold"],
        path + ".closure_catchment_shock_hold");
  }
  if (has_key(band_ale, "closure_catchment_eta_h_arm")) {
    config.closure_catchment_eta_h_arm = numeric_as_double(
        band_ale["closure_catchment_eta_h_arm"],
        path + ".closure_catchment_eta_h_arm");
  }
  if (has_key(band_ale, "closure_catchment_eta_h_full")) {
    config.closure_catchment_eta_h_full = numeric_as_double(
        band_ale["closure_catchment_eta_h_full"],
        path + ".closure_catchment_eta_h_full");
  }
  if (has_key(band_ale, "closure_catchment_eta_m_arm")) {
    config.closure_catchment_eta_m_arm = numeric_as_double(
        band_ale["closure_catchment_eta_m_arm"],
        path + ".closure_catchment_eta_m_arm");
  }
  if (has_key(band_ale, "closure_catchment_eta_m_full")) {
    config.closure_catchment_eta_m_full = numeric_as_double(
        band_ale["closure_catchment_eta_m_full"],
        path + ".closure_catchment_eta_m_full");
  }
  if (has_key(band_ale, "closure_catchment_reset_eta")) {
    config.closure_catchment_reset_eta = numeric_as_double(
        band_ale["closure_catchment_reset_eta"],
        path + ".closure_catchment_reset_eta");
  }
  if (has_key(band_ale, "closure_catchment_support_core_rows")) {
    config.closure_catchment_support_core_rows = strict_int32(
        band_ale["closure_catchment_support_core_rows"],
        path + ".closure_catchment_support_core_rows");
  }
  if (has_key(band_ale, "closure_catchment_support_taper_rows")) {
    config.closure_catchment_support_taper_rows = strict_int32(
        band_ale["closure_catchment_support_taper_rows"],
        path + ".closure_catchment_support_taper_rows");
  }
  if (has_key(band_ale, "closure_catchment_accum_frac")) {
    config.closure_catchment_accum_frac = numeric_as_double(
        band_ale["closure_catchment_accum_frac"],
        path + ".closure_catchment_accum_frac");
  }
  if (has_key(band_ale, "closure_catchment_rearm_drop")) {
    config.closure_catchment_rearm_drop = numeric_as_double(
        band_ale["closure_catchment_rearm_drop"],
        path + ".closure_catchment_rearm_drop");
  }
  if (has_key(band_ale, "pole_theta_enabled")) {
    config.pole_theta_enabled = strict_bool(
        band_ale["pole_theta_enabled"], path + ".pole_theta_enabled");
  }
  if (has_key(band_ale, "pole_theta_routine_enabled")) {
    config.pole_theta_routine_enabled = strict_bool(
        band_ale["pole_theta_routine_enabled"],
        path + ".pole_theta_routine_enabled");
  }
  if (has_key(band_ale, "pole_theta_phys_lp")) {
    config.pole_theta_phys_lp = strict_int32(
        band_ale["pole_theta_phys_lp"], path + ".pole_theta_phys_lp");
  }
  if (has_key(band_ale, "pole_theta_phys_lc")) {
    config.pole_theta_phys_lc = strict_int32(
        band_ale["pole_theta_phys_lc"], path + ".pole_theta_phys_lc");
  }
  if (has_key(band_ale, "pole_theta_noise_floor")) {
    config.pole_theta_noise_floor = numeric_as_double(
        band_ale["pole_theta_noise_floor"],
        path + ".pole_theta_noise_floor");
  }
  if (has_key(band_ale, "pole_theta_noise_ceiling")) {
    config.pole_theta_noise_ceiling = numeric_as_double(
        band_ale["pole_theta_noise_ceiling"],
        path + ".pole_theta_noise_ceiling");
  }
  if (has_key(band_ale, "pole_theta_h_arm")) {
    config.pole_theta_h_arm = numeric_as_double(
        band_ale["pole_theta_h_arm"], path + ".pole_theta_h_arm");
  }
  if (has_key(band_ale, "pole_theta_h_fire")) {
    config.pole_theta_h_fire = numeric_as_double(
        band_ale["pole_theta_h_fire"], path + ".pole_theta_h_fire");
  }
  if (has_key(band_ale, "pole_theta_h_hard")) {
    config.pole_theta_h_hard = numeric_as_double(
        band_ale["pole_theta_h_hard"], path + ".pole_theta_h_hard");
  }
  if (has_key(band_ale, "pole_theta_h_release")) {
    config.pole_theta_h_release = numeric_as_double(
        band_ale["pole_theta_h_release"], path + ".pole_theta_h_release");
  }
  if (has_key(band_ale, "pole_theta_kappa_arm")) {
    config.pole_theta_kappa_arm = numeric_as_double(
        band_ale["pole_theta_kappa_arm"], path + ".pole_theta_kappa_arm");
  }
  if (has_key(band_ale, "pole_theta_kappa_fire")) {
    config.pole_theta_kappa_fire = numeric_as_double(
        band_ale["pole_theta_kappa_fire"],
        path + ".pole_theta_kappa_fire");
  }
  if (has_key(band_ale, "pole_theta_kappa_hard")) {
    config.pole_theta_kappa_hard = numeric_as_double(
        band_ale["pole_theta_kappa_hard"],
        path + ".pole_theta_kappa_hard");
  }
  if (has_key(band_ale, "pole_theta_alpha")) {
    config.pole_theta_alpha = numeric_as_double(
        band_ale["pole_theta_alpha"], path + ".pole_theta_alpha");
  }
  if (has_key(band_ale, "pole_theta_alpha_hard")) {
    config.pole_theta_alpha_hard = numeric_as_double(
        band_ale["pole_theta_alpha_hard"], path + ".pole_theta_alpha_hard");
  }
  if (has_key(band_ale, "pole_theta_deadband_frac")) {
    config.pole_theta_deadband_frac = numeric_as_double(
        band_ale["pole_theta_deadband_frac"],
        path + ".pole_theta_deadband_frac");
  }
  if (has_key(band_ale, "pole_theta_move_limit_frac")) {
    config.pole_theta_move_limit_frac = numeric_as_double(
        band_ale["pole_theta_move_limit_frac"],
        path + ".pole_theta_move_limit_frac");
  }
  if (has_key(band_ale, "pole_theta_move_limit_hard_frac")) {
    config.pole_theta_move_limit_hard_frac = numeric_as_double(
        band_ale["pole_theta_move_limit_hard_frac"],
        path + ".pole_theta_move_limit_hard_frac");
  }
  if (has_key(band_ale, "pole_theta_cooldown_s")) {
    config.pole_theta_cooldown_s = numeric_as_double(
        band_ale["pole_theta_cooldown_s"], path + ".pole_theta_cooldown_s");
  }
  if (has_key(band_ale, "pole_theta_cooldown_base_s")) {
    config.pole_theta_cooldown_base_s = numeric_as_double(
        band_ale["pole_theta_cooldown_base_s"],
        path + ".pole_theta_cooldown_base_s");
  }
  if (has_key(band_ale, "pole_theta_predict_window_s")) {
    config.pole_theta_predict_window_s = numeric_as_double(
        band_ale["pole_theta_predict_window_s"],
        path + ".pole_theta_predict_window_s");
  }
  if (has_key(band_ale, "pole_theta_predict_horizon_s")) {
    config.pole_theta_predict_horizon_s = numeric_as_double(
        band_ale["pole_theta_predict_horizon_s"],
        path + ".pole_theta_predict_horizon_s");
  }
  if (has_key(band_ale, "pole_theta_halo_columns")) {
    config.pole_theta_halo_columns = strict_int32(
        band_ale["pole_theta_halo_columns"],
        path + ".pole_theta_halo_columns");
  }
  if (has_key(band_ale, "pole_theta_halo_rows")) {
    config.pole_theta_halo_rows = strict_int32(
        band_ale["pole_theta_halo_rows"], path + ".pole_theta_halo_rows");
  }
  if (has_key(band_ale, "pole_theta_post_h_floor")) {
    config.pole_theta_post_h_floor = numeric_as_double(
        band_ale["pole_theta_post_h_floor"],
        path + ".pole_theta_post_h_floor");
  }
  if (has_key(band_ale, "pole_theta_curve_preserving")) {
    config.pole_theta_curve_preserving = strict_bool(
        band_ale["pole_theta_curve_preserving"],
        path + ".pole_theta_curve_preserving");
  }
  if (has_key(band_ale, "pole_theta_protected_modes")) {
    config.pole_theta_protected_modes = strict_string(
        band_ale["pole_theta_protected_modes"],
        path + ".pole_theta_protected_modes");
  }
  if (has_key(band_ale, "pole_theta_fit_order")) {
    config.pole_theta_fit_order = strict_int32(
        band_ale["pole_theta_fit_order"],
        path + ".pole_theta_fit_order");
  }
  if (has_key(band_ale, "pole_theta_shock_hold")) {
    config.pole_theta_shock_hold = numeric_as_double(
        band_ale["pole_theta_shock_hold"],
        path + ".pole_theta_shock_hold");
  }
  if (has_key(band_ale, "shell_window_in_rows")) {
    config.shell_window_in_rows = strict_int32(
        band_ale["shell_window_in_rows"],
        path + ".shell_window_in_rows");
  }
  if (has_key(band_ale, "shell_window_out_rows")) {
    config.shell_window_out_rows = strict_int32(
        band_ale["shell_window_out_rows"],
        path + ".shell_window_out_rows");
  }
  if (has_key(band_ale, "shell_boundary_guard_rows")) {
    config.shell_boundary_guard_rows = strict_int32(
        band_ale["shell_boundary_guard_rows"],
        path + ".shell_boundary_guard_rows");
  }
  if (has_key(band_ale, "shell_min_spacing_frac")) {
    config.shell_min_spacing_frac = numeric_as_double(
        band_ale["shell_min_spacing_frac"],
        path + ".shell_min_spacing_frac");
  }
  if (has_key(band_ale, "shell_front_metric")) {
    config.shell_front_metric = strict_string(
        band_ale["shell_front_metric"], path + ".shell_front_metric");
  }
  if (has_key(band_ale, "shell_target")) {
    config.shell_target =
        strict_string(band_ale["shell_target"], path + ".shell_target");
  }
  if (has_key(band_ale, "axis_repair_enabled")) {
    config.axis_repair_enabled = strict_bool(
        band_ale["axis_repair_enabled"], path + ".axis_repair_enabled");
  }
  if (has_key(band_ale, "axis_repair_eta_on")) {
    config.axis_repair_eta_on = numeric_as_double(
        band_ale["axis_repair_eta_on"], path + ".axis_repair_eta_on");
  }
  if (has_key(band_ale, "axis_repair_eta_off")) {
    config.axis_repair_eta_off = numeric_as_double(
        band_ale["axis_repair_eta_off"], path + ".axis_repair_eta_off");
  }
  if (has_key(band_ale, "axis_repair_cap_rel")) {
    config.axis_repair_cap_rel = numeric_as_double(
        band_ale["axis_repair_cap_rel"], path + ".axis_repair_cap_rel");
  }
}

void parse_evacuated_cell_dict(
    const py::dict& evacuated_cell,
    const std::string& path,
    Config::NumericsConfig::AleConfig::EvacuatedCellConfig& config) {
  enforce_known_keys(
      evacuated_cell,
      path,
      {"enabled",
       "every_n_steps",
       "arm_mass_fraction",
       "off_mass_fraction",
       "rho_vacuum_policy_g_per_cc",
       "off_hold_evaluations",
       "laser_ne_over_ncrit_max",
       "laser_wavelength_nm",
       "coupling_fraction_max",
       "max_cells_per_event",
       "rematerialize_enabled",
       "rematerialize_after_evaluations",
       "rematerialize_volume_fraction",
       "rematerialize_neighbor_change_max",
       "rematerialize_dwell_evaluations",
       "closure_contact"});
  if (has_key(evacuated_cell, "enabled")) {
    config.enabled =
        strict_bool(evacuated_cell["enabled"], path + ".enabled");
  }
  if (has_key(evacuated_cell, "every_n_steps")) {
    config.every_n_steps = strict_int32(
        evacuated_cell["every_n_steps"], path + ".every_n_steps");
  }
  if (has_key(evacuated_cell, "arm_mass_fraction")) {
    config.arm_mass_fraction = numeric_as_double(
        evacuated_cell["arm_mass_fraction"], path + ".arm_mass_fraction");
  }
  if (has_key(evacuated_cell, "off_mass_fraction")) {
    config.off_mass_fraction = numeric_as_double(
        evacuated_cell["off_mass_fraction"], path + ".off_mass_fraction");
  }
  if (has_key(evacuated_cell, "rho_vacuum_policy_g_per_cc")) {
    config.rho_vacuum_policy_g_per_cc = numeric_as_double(
        evacuated_cell["rho_vacuum_policy_g_per_cc"],
        path + ".rho_vacuum_policy_g_per_cc");
  }
  if (has_key(evacuated_cell, "off_hold_evaluations")) {
    config.off_hold_evaluations = strict_int32(
        evacuated_cell["off_hold_evaluations"],
        path + ".off_hold_evaluations");
  }
  if (has_key(evacuated_cell, "laser_ne_over_ncrit_max")) {
    config.laser_ne_over_ncrit_max = numeric_as_double(
        evacuated_cell["laser_ne_over_ncrit_max"],
        path + ".laser_ne_over_ncrit_max");
  }
  if (has_key(evacuated_cell, "laser_wavelength_nm")) {
    config.laser_wavelength_nm = numeric_as_double(
        evacuated_cell["laser_wavelength_nm"],
        path + ".laser_wavelength_nm");
  }
  if (has_key(evacuated_cell, "coupling_fraction_max")) {
    config.coupling_fraction_max = numeric_as_double(
        evacuated_cell["coupling_fraction_max"],
        path + ".coupling_fraction_max");
  }
  if (has_key(evacuated_cell, "max_cells_per_event")) {
    config.max_cells_per_event = strict_int32(
        evacuated_cell["max_cells_per_event"],
        path + ".max_cells_per_event");
  }
  if (has_key(evacuated_cell, "rematerialize_enabled")) {
    config.rematerialize_enabled = strict_bool(
        evacuated_cell["rematerialize_enabled"],
        path + ".rematerialize_enabled");
  }
  if (has_key(evacuated_cell, "rematerialize_after_evaluations")) {
    config.rematerialize_after_evaluations = strict_int32(
        evacuated_cell["rematerialize_after_evaluations"],
        path + ".rematerialize_after_evaluations");
  }
  if (has_key(evacuated_cell, "rematerialize_volume_fraction")) {
    config.rematerialize_volume_fraction = numeric_as_double(
        evacuated_cell["rematerialize_volume_fraction"],
        path + ".rematerialize_volume_fraction");
  }
  if (has_key(evacuated_cell, "rematerialize_neighbor_change_max")) {
    config.rematerialize_neighbor_change_max = numeric_as_double(
        evacuated_cell["rematerialize_neighbor_change_max"],
        path + ".rematerialize_neighbor_change_max");
  }
  if (has_key(evacuated_cell, "rematerialize_dwell_evaluations")) {
    config.rematerialize_dwell_evaluations = strict_int32(
        evacuated_cell["rematerialize_dwell_evaluations"],
        path + ".rematerialize_dwell_evaluations");
  }
  if (has_key(evacuated_cell, "closure_contact")) {
    const py::handle closure_contact_obj = evacuated_cell["closure_contact"];
    if (!py::isinstance<py::dict>(closure_contact_obj)) {
      throw_value_type_error(
          path + ".closure_contact", "dict", closure_contact_obj);
    }
    const py::dict closure_contact =
        py::reinterpret_borrow<py::dict>(closure_contact_obj);
    const std::string closure_path = path + ".closure_contact";
    enforce_known_keys(
        closure_contact,
        closure_path,
        {"enabled",
         "gap_floor_fraction",
         "gap_arm_fraction",
         "live_mass_gate",
         "live_volume_gate",
         "refill_min_mass_fraction",
         "refill_min_density_ratio",
         "release_force_c",
         "release_persistence_stages",
         "reengage_gap_margin",
         "mortar_position_drift_beta",
         "surface_engage_enabled",
         "lcp_apply_enabled",
         "axis_edge_collapse",
         "flank_tangential_strip",
         "seam_interface_owner_enabled"});
    auto& closure = config.closure_contact;
    if (has_key(closure_contact, "enabled")) {
      closure.enabled = strict_bool(
          closure_contact["enabled"], closure_path + ".enabled");
    }
    if (has_key(closure_contact, "gap_floor_fraction")) {
      closure.gap_floor_fraction = numeric_as_double(
          closure_contact["gap_floor_fraction"],
          closure_path + ".gap_floor_fraction");
    }
    if (has_key(closure_contact, "gap_arm_fraction")) {
      closure.gap_arm_fraction = numeric_as_double(
          closure_contact["gap_arm_fraction"],
          closure_path + ".gap_arm_fraction");
    }
    if (has_key(closure_contact, "live_mass_gate")) {
      closure.live_mass_gate = numeric_as_double(
          closure_contact["live_mass_gate"],
          closure_path + ".live_mass_gate");
    }
    if (has_key(closure_contact, "live_volume_gate")) {
      closure.live_volume_gate = numeric_as_double(
          closure_contact["live_volume_gate"],
          closure_path + ".live_volume_gate");
    }
    if (has_key(closure_contact, "refill_min_mass_fraction")) {
      closure.refill_min_mass_fraction = numeric_as_double(
          closure_contact["refill_min_mass_fraction"],
          closure_path + ".refill_min_mass_fraction");
    }
    if (has_key(closure_contact, "refill_min_density_ratio")) {
      closure.refill_min_density_ratio = numeric_as_double(
          closure_contact["refill_min_density_ratio"],
          closure_path + ".refill_min_density_ratio");
    }
    if (has_key(closure_contact, "release_force_c")) {
      closure.release_force_c = numeric_as_double(
          closure_contact["release_force_c"],
          closure_path + ".release_force_c");
    }
    if (has_key(closure_contact, "release_persistence_stages")) {
      closure.release_persistence_stages = strict_int32(
          closure_contact["release_persistence_stages"],
          closure_path + ".release_persistence_stages");
    }
    if (has_key(closure_contact, "reengage_gap_margin")) {
      closure.reengage_gap_margin = numeric_as_double(
          closure_contact["reengage_gap_margin"],
          closure_path + ".reengage_gap_margin");
    }
    if (has_key(closure_contact, "mortar_position_drift_beta")) {
      closure.mortar_position_drift_beta = numeric_as_double(
          closure_contact["mortar_position_drift_beta"],
          closure_path + ".mortar_position_drift_beta");
    }
    if (has_key(closure_contact, "surface_engage_enabled")) {
      closure.surface_engage_enabled = strict_bool(
          closure_contact["surface_engage_enabled"],
          closure_path + ".surface_engage_enabled");
    }
    if (has_key(closure_contact, "lcp_apply_enabled")) {
      closure.lcp_apply_enabled = strict_bool(
          closure_contact["lcp_apply_enabled"],
          closure_path + ".lcp_apply_enabled");
    }
    if (has_key(closure_contact, "axis_edge_collapse")) {
      const py::handle axis_edge_collapse_obj =
          closure_contact["axis_edge_collapse"];
      if (!py::isinstance<py::dict>(axis_edge_collapse_obj)) {
        throw_value_type_error(closure_path + ".axis_edge_collapse",
                               "dict",
                               axis_edge_collapse_obj);
      }
      const py::dict axis_edge_collapse =
          py::reinterpret_borrow<py::dict>(axis_edge_collapse_obj);
      const std::string collapse_path =
          closure_path + ".axis_edge_collapse";
      enforce_known_keys(axis_edge_collapse,
                         collapse_path,
                         {"enabled",
                          "ulp_count",
                          "h_ref_fraction",
                          "release_hysteresis",
                          "persistence_window",
                          "persistence_min_closing",
                          "repair_recurrence_steps",
                          "repair_futility_fraction"});
      auto& collapse = closure.axis_edge_collapse;
      if (has_key(axis_edge_collapse, "enabled")) {
        collapse.enabled = strict_bool(
            axis_edge_collapse["enabled"], collapse_path + ".enabled");
      }
      if (has_key(axis_edge_collapse, "ulp_count")) {
        collapse.ulp_count = numeric_as_double(
            axis_edge_collapse["ulp_count"], collapse_path + ".ulp_count");
      }
      if (has_key(axis_edge_collapse, "h_ref_fraction")) {
        collapse.h_ref_fraction = numeric_as_double(
            axis_edge_collapse["h_ref_fraction"],
            collapse_path + ".h_ref_fraction");
      }
      if (has_key(axis_edge_collapse, "release_hysteresis")) {
        collapse.release_hysteresis = numeric_as_double(
            axis_edge_collapse["release_hysteresis"],
            collapse_path + ".release_hysteresis");
      }
      if (has_key(axis_edge_collapse, "persistence_window")) {
        collapse.persistence_window = strict_int32(
            axis_edge_collapse["persistence_window"],
            collapse_path + ".persistence_window");
      }
      if (has_key(axis_edge_collapse, "persistence_min_closing")) {
        collapse.persistence_min_closing = strict_int32(
            axis_edge_collapse["persistence_min_closing"],
            collapse_path + ".persistence_min_closing");
      }
      if (has_key(axis_edge_collapse, "repair_recurrence_steps")) {
        collapse.repair_recurrence_steps = strict_int32(
            axis_edge_collapse["repair_recurrence_steps"],
            collapse_path + ".repair_recurrence_steps");
      }
      if (has_key(axis_edge_collapse, "repair_futility_fraction")) {
        collapse.repair_futility_fraction = numeric_as_double(
            axis_edge_collapse["repair_futility_fraction"],
            collapse_path + ".repair_futility_fraction");
      }
    }
    if (has_key(closure_contact, "flank_tangential_strip")) {
      const py::handle flank_tangential_strip_obj =
          closure_contact["flank_tangential_strip"];
      if (!py::isinstance<py::dict>(flank_tangential_strip_obj)) {
        throw_value_type_error(closure_path + ".flank_tangential_strip",
                               "dict",
                               flank_tangential_strip_obj);
      }
      const py::dict flank_tangential_strip =
          py::reinterpret_borrow<py::dict>(flank_tangential_strip_obj);
      const std::string strip_path =
          closure_path + ".flank_tangential_strip";
      enforce_known_keys(flank_tangential_strip,
                         strip_path,
                         {"enabled",
                          "untangler_enabled",
                          "band_layers",
                          "band_halfwidth_j",
                          "arm_quality_ratio",
                          "release_quality_ratio",
                          "min_progress_factor",
                          "lead_steps",
                          "release_persistence_steps",
                          "release_shear_number",
                          "slip_handoff_ratio",
                          "slip_patch_enabled"});
      auto& strip = closure.flank_tangential_strip;
      if (has_key(flank_tangential_strip, "enabled")) {
        strip.enabled = strict_bool(
            flank_tangential_strip["enabled"], strip_path + ".enabled");
      }
      if (has_key(flank_tangential_strip, "untangler_enabled")) {
        strip.untangler_enabled = strict_bool(
            flank_tangential_strip["untangler_enabled"],
            strip_path + ".untangler_enabled");
      }
      if (has_key(flank_tangential_strip, "band_layers")) {
        strip.band_layers = strict_int32(
            flank_tangential_strip["band_layers"],
            strip_path + ".band_layers");
      }
      if (has_key(flank_tangential_strip, "band_halfwidth_j")) {
        strip.band_halfwidth_j = strict_int32(
            flank_tangential_strip["band_halfwidth_j"],
            strip_path + ".band_halfwidth_j");
      }
      if (has_key(flank_tangential_strip, "arm_quality_ratio")) {
        strip.arm_quality_ratio = numeric_as_double(
            flank_tangential_strip["arm_quality_ratio"],
            strip_path + ".arm_quality_ratio");
      }
      if (has_key(flank_tangential_strip, "release_quality_ratio")) {
        strip.release_quality_ratio = numeric_as_double(
            flank_tangential_strip["release_quality_ratio"],
            strip_path + ".release_quality_ratio");
      }
      if (has_key(flank_tangential_strip, "min_progress_factor")) {
        strip.min_progress_factor = numeric_as_double(
            flank_tangential_strip["min_progress_factor"],
            strip_path + ".min_progress_factor");
      }
      if (has_key(flank_tangential_strip, "lead_steps")) {
        strip.lead_steps = strict_int32(
            flank_tangential_strip["lead_steps"],
            strip_path + ".lead_steps");
      }
      if (has_key(flank_tangential_strip,
                  "release_persistence_steps")) {
        strip.release_persistence_steps = strict_int32(
            flank_tangential_strip["release_persistence_steps"],
            strip_path + ".release_persistence_steps");
      }
      if (has_key(flank_tangential_strip, "release_shear_number")) {
        strip.release_shear_number = numeric_as_double(
            flank_tangential_strip["release_shear_number"],
            strip_path + ".release_shear_number");
      }
      if (has_key(flank_tangential_strip, "slip_handoff_ratio")) {
        strip.slip_handoff_ratio = numeric_as_double(
            flank_tangential_strip["slip_handoff_ratio"],
            strip_path + ".slip_handoff_ratio");
      }
      if (has_key(flank_tangential_strip, "slip_patch_enabled")) {
        strip.slip_patch_enabled = strict_bool(
            flank_tangential_strip["slip_patch_enabled"],
            strip_path + ".slip_patch_enabled");
      }
    }
    if (has_key(closure_contact, "seam_interface_owner_enabled")) {
      closure.seam_interface_owner_enabled = strict_bool(
          closure_contact["seam_interface_owner_enabled"],
          closure_path + ".seam_interface_owner_enabled");
    }
  }
}

void validate_band_ale_config(
    const Config::NumericsConfig::AleConfig::BandAleConfig& config,
    const std::string& path) {
  if (config.estimator_band_per_column && config.bands != "estimator") {
    throw ConfigError(
        path + ".estimator_band_per_column=true requires bands=\"estimator\"");
  }
  if (!config.enabled) {
    return;
  }
  if (!(std::isfinite(config.aspect_trigger) &&
        config.aspect_trigger > 0.0 && config.aspect_trigger <= 1.0)) {
    throw ValueError(path + ".aspect_trigger must be finite and in (0, 1]");
  }
  if (!(std::isfinite(config.release_hysteresis) &&
        config.release_hysteresis > 1.0)) {
    throw ValueError(
        path + ".release_hysteresis must be finite and > 1");
  }
  if (!(std::isfinite(config.chi) &&
        config.chi >= 0.0 && config.chi <= 1.0)) {
    throw ValueError(path + ".chi must be finite and in [0, 1]");
  }
  if (config.bands != "belts" && config.bands != "axis" &&
      config.bands != "belts_axis" && config.bands != "shell" &&
      config.bands != "belts_axis_shell" && config.bands != "estimator") {
    throw ValueError(
        path + ".bands must be one of {\"belts\", \"axis\", "
               "\"belts_axis\", \"shell\", \"belts_axis_shell\", "
               "\"estimator\"}");
  }
  if (config.belt_target != "ring_mean" &&
      config.belt_target != "respace") {
    throw ValueError(
        path + ".belt_target must be one of {\"ring_mean\", \"respace\"}");
  }
  if (config.center_target != "line" &&
      config.center_target != "volume_fraction") {
    throw ValueError(
        path +
        ".center_target must be one of {\"line\", \"volume_fraction\"}");
  }
  if (config.axis_target != "z_laplacian" &&
      config.axis_target != "respace") {
    throw ValueError(
        path +
        ".axis_target must be one of {\"z_laplacian\", \"respace\"}");
  }
  if (config.axis_segment_halfwidth < 1 ||
      config.axis_segment_halfwidth > 64) {
    throw ValueError(
        path + ".axis_segment_halfwidth must be in [1, 64]");
  }
  if (!(std::isfinite(config.estimator_band_cut) &&
        config.estimator_band_cut > 0.0 &&
        config.estimator_band_cut < 1.0)) {
    throw ValueError(
        path + ".estimator_band_cut must be finite and in (0, 1)");
  }
  if (!(std::isfinite(config.estimator_band_shock_hold) &&
        config.estimator_band_shock_hold > config.estimator_band_cut &&
        config.estimator_band_shock_hold <= 1.0)) {
    throw ValueError(
        path +
        ".estimator_band_shock_hold must be finite, greater than "
        "estimator_band_cut, and <= 1.0");
  }
  if (!(config.estimator_band_front_hold_margin_rows >= 0.0)) {
    throw ConfigError(
        path + ".estimator_band_front_hold_margin_rows must be >= 0.0");
  }
  if (config.estimator_band_axis != "auto" &&
      config.estimator_band_axis != "i" &&
      config.estimator_band_axis != "j") {
    throw ConfigError(
        path + ".estimator_band_axis must be one of {\"auto\", \"i\", \"j\"}");
  }
  if (config.estimator_band_in_rows < 1) {
    throw ValueError(path + ".estimator_band_in_rows must be >= 1");
  }
  if (config.estimator_band_out_rows < 1) {
    throw ValueError(path + ".estimator_band_out_rows must be >= 1");
  }
  if (!(std::isfinite(config.estimator_band_eta_on) &&
        std::isfinite(config.estimator_band_eta_off) &&
        config.estimator_band_eta_off > 1.0 &&
        config.estimator_band_eta_off < config.estimator_band_eta_on)) {
    throw ValueError(
        path +
        ".estimator_band_eta_on and estimator_band_eta_off must satisfy "
        "1 < eta_off < eta_on");
  }
  if (config.estimator_band_pc_filter_halfwidth < 1 ||
      config.estimator_band_pc_filter_halfwidth > 3) {
    throw ValueError(
        path + ".estimator_band_pc_filter_halfwidth must be in [1, 3]");
  }
  if (!(std::isfinite(config.estimator_band_pc_slope_limit) &&
        config.estimator_band_pc_slope_limit > 0.0)) {
    throw ValueError(
        path + ".estimator_band_pc_slope_limit must be finite and > 0");
  }
  if (!(std::isfinite(config.estimator_band_pc_slope_reject) &&
        config.estimator_band_pc_slope_reject >=
            config.estimator_band_pc_slope_limit)) {
    throw ValueError(
        path + ".estimator_band_pc_slope_reject must be finite and >= "
        "estimator_band_pc_slope_limit");
  }
  if (!(std::isfinite(config.estimator_band_pc_curvature_limit) &&
        config.estimator_band_pc_curvature_limit > 0.0)) {
    throw ValueError(
        path + ".estimator_band_pc_curvature_limit must be finite and > 0");
  }
  if (!(std::isfinite(config.estimator_band_pc_chi_max) &&
        config.estimator_band_pc_chi_max > 0.0 &&
        config.estimator_band_pc_chi_max <= 1.0)) {
    throw ValueError(
        path + ".estimator_band_pc_chi_max must be finite and in (0, 1]");
  }
  if (!(std::isfinite(config.estimator_band_pc_chi_step) &&
        config.estimator_band_pc_chi_step > 0.0)) {
    throw ValueError(
        path + ".estimator_band_pc_chi_step must be finite and > 0");
  }
  if (!(std::isfinite(config.estimator_band_pc_sigma_floor) &&
        config.estimator_band_pc_sigma_floor > 0.0 &&
        config.estimator_band_pc_sigma_floor < 1.0)) {
    throw ValueError(
        path + ".estimator_band_pc_sigma_floor must be finite and in (0, 1)");
  }
  if (!(std::isfinite(config.estimator_band_pc_coverage_min) &&
        std::isfinite(config.estimator_band_pc_coverage_full) &&
        config.estimator_band_pc_coverage_min > 0.0 &&
        config.estimator_band_pc_coverage_min <=
            config.estimator_band_pc_coverage_full &&
        config.estimator_band_pc_coverage_full <= 1.0)) {
    throw ValueError(
        path + ".estimator_band_pc_coverage_min and "
        "estimator_band_pc_coverage_full must satisfy "
        "0 < coverage_min <= coverage_full <= 1");
  }
  if (config.estimator_band_pc_cooldown_events < 1) {
    throw ValueError(
        path + ".estimator_band_pc_cooldown_events must be >= 1");
  }
  if (config.estimator_band_pc_tube_dilate_rows < 0) {
    throw ValueError(path + ".estimator_band_pc_tube_dilate_rows must be >= 0");
  }
  if (config.estimator_band_pc_tube_dilate_cols_extra < 0) {
    throw ValueError(path + ".estimator_band_pc_tube_dilate_cols_extra must be >= 0");
  }
  if (!(std::isfinite(config.estimator_band_pc_ambiguous_hold_fraction) &&
        config.estimator_band_pc_ambiguous_hold_fraction >= 0.0 &&
        config.estimator_band_pc_ambiguous_hold_fraction <= 1.0)) {
    throw ValueError(
        path + ".estimator_band_pc_ambiguous_hold_fraction must be finite and in [0, 1]");
  }
  if (config.shell_window_in_rows < 0) {
    throw ValueError(path + ".shell_window_in_rows must be >= 0");
  }
  if (config.shell_window_out_rows < 0) {
    throw ValueError(path + ".shell_window_out_rows must be >= 0");
  }
  if (config.shell_boundary_guard_rows < 0) {
    throw ValueError(path + ".shell_boundary_guard_rows must be >= 0");
  }
  if (config.shell_front_metric != "grad_rho" &&
      config.shell_front_metric != "min_spacing") {
    throw ValueError(
        path +
        ".shell_front_metric must be one of {\"grad_rho\", \"min_spacing\"}");
  }
  if (config.shell_target != "respace") {
    throw ValueError(path + ".shell_target must be \"respace\"");
  }
  if (!(std::isfinite(config.axis_repair_eta_on) &&
        std::isfinite(config.axis_repair_eta_off) &&
        config.axis_repair_eta_on > 0.0 &&
        config.axis_repair_eta_on < config.axis_repair_eta_off &&
        config.axis_repair_eta_off <= 1.0)) {
    throw ValueError(
        path +
        ".axis_repair_eta_on and axis_repair_eta_off must satisfy "
        "0 < eta_on < eta_off <= 1");
  }
  if (!(std::isfinite(config.axis_repair_cap_rel) &&
        config.axis_repair_cap_rel > 0.0 &&
        config.axis_repair_cap_rel <= 0.2)) {
    throw ValueError(
        path + ".axis_repair_cap_rel must be finite and in (0, 0.2]");
  }
}

void warn_ignored_key(std::string_view key_path) {
  tenryu::core::log_warning(std::string(key_path) +
                            " is accepted for compatibility and ignored in M01");
}

std::string normalize_boundary_mode(std::string mode, std::string_view key_path) {
  if (mode == "reflective") {
    tenryu::core::log_warning(std::string(key_path) +
                              "='reflective' is deprecated; using 'reflect'");
    return "reflect";
  }
  return mode;
}

bool is_dimension(const std::string& value) {
  return value == "1D_SPH" || value == "1D_CYL" || value == "2D_RZ";
}

bool is_dimension_1d(const std::string& value) {
  return value == "1D_SPH" || value == "1D_CYL";
}

bool is_verbosity(const std::string& value) {
  return value == "quiet" || value == "normal" || value == "verbose" ||
         value == "debug";
}

bool is_temperature_model(const std::string& value) {
  return value == "1T" || value == "2T" || value == "auto";
}

bool is_grid_type(const std::string& value) {
  return value == "uniform" || value == "graded";
}

void parse_mesh_grid_segments_into(
    const py::handle segments_obj,
    const char* context,
    std::vector<Config::MeshConfig::GridSegment>& out,
    std::string& out_repr) {
  const std::string context_str(context);
  if (!py::isinstance<py::sequence>(segments_obj) || py::isinstance<py::str>(segments_obj)) {
    throw_value_type_error(context_str, "list[dict]", segments_obj);
  }
  const auto seg_seq = py::reinterpret_borrow<py::sequence>(segments_obj);
  out_repr = py::repr(segments_obj).cast<std::string>();
  out.clear();
  out.reserve(seg_seq.size());
  for (std::size_t i = 0; i < seg_seq.size(); ++i) {
    const py::handle seg_obj = seg_seq[i];
    if (!py::isinstance<py::dict>(seg_obj)) {
      throw_value_type_error(context_str + "[" + std::to_string(i) + "]",
                             "dict",
                             seg_obj);
    }
    const py::dict seg = py::reinterpret_borrow<py::dict>(seg_obj);
    enforce_known_keys(seg, context_str, {"r_start", "r_end", "nr"});
    if (!has_key(seg, "r_start")) {
      throw ConfigError(context_str + "[" + std::to_string(i) +
                        "] requires key 'r_start'");
    }
    if (!has_key(seg, "r_end")) {
      throw ConfigError(context_str + "[" + std::to_string(i) +
                        "] requires key 'r_end'");
    }
    if (!has_key(seg, "nr")) {
      throw ConfigError(context_str + "[" + std::to_string(i) +
                        "] requires key 'nr'");
    }

    Config::MeshConfig::GridSegment gs;
    gs.r_start =
        numeric_as_double(seg["r_start"], context_str + "[" + std::to_string(i) + "].r_start");
    gs.r_end =
        numeric_as_double(seg["r_end"], context_str + "[" + std::to_string(i) + "].r_end");
    gs.nr = strict_int32(seg["nr"], context_str + "[" + std::to_string(i) + "].nr");
    if (gs.nr <= 0) {
      throw ValueError(context_str + ": each segment.nr must be > 0");
    }
    if (gs.r_end <= gs.r_start) {
      throw ValueError(context_str +
                       ": r_end must be > r_start in each segment");
    }
    out.push_back(gs);
  }

  if (out.empty()) {
    throw ConfigError(context_str + " must be a non-empty list");
  }
  for (std::size_t k = 1; k < out.size(); ++k) {
    const double gap = std::abs(out[k].r_start - out[k - 1].r_end);
    const double tol = 1.0e-12 * std::max(1.0, out[k - 1].r_end);
    if (gap > tol) {
      throw ConfigError(context_str + ": gap between segment " +
                        std::to_string(k - 1) + " and " + std::to_string(k));
    }
  }
}

void parse_mesh_grid_segments(const py::handle segments_obj,
                              Config::MeshConfig& mesh) {
  parse_mesh_grid_segments_into(segments_obj, "Mesh.grid.segments",
                                mesh.grid_segments, mesh.grid_segments_repr);
  int nr_sum = 0;
  for (const auto& gs : mesh.grid_segments) {
    nr_sum += gs.nr;
  }
  if (mesh.nr > 0 && mesh.nr != nr_sum) {
    tenryu::core::log_warning("Mesh.nr=" + std::to_string(mesh.nr) +
                              " overridden by grid segments sum=" +
                              std::to_string(nr_sum));
  }
  mesh.nr = nr_sum;
  tenryu::core::log_info("Mesh.grid.segments parsed: " +
                         std::to_string(mesh.grid_segments.size()) +
                         " segments, total nr=" + std::to_string(nr_sum));
}

void parse_mesh_grid_segments_z(const py::handle segments_obj,
                                Config::MeshConfig& mesh) {
  parse_mesh_grid_segments_into(segments_obj, "Mesh.grid_z.segments",
                                mesh.grid_segments_z,
                                mesh.grid_segments_z_repr);
  int nz_sum = 0;
  for (const auto& gs : mesh.grid_segments_z) {
    nz_sum += gs.nr;
  }
  if (mesh.nz > 1 && mesh.nz != nz_sum) {
    throw ConfigError("Mesh.nz (" + std::to_string(mesh.nz) +
                      ") conflicts with Mesh.grid_z segments total (" +
                      std::to_string(nz_sum) +
                      "); omit Mesh.nz (or leave it 1) or match it");
  }
  mesh.nz = nz_sum;
  tenryu::core::log_info("Mesh.grid_z.segments parsed: " +
                         std::to_string(mesh.grid_segments_z.size()) +
                         " segments, total nz=" + std::to_string(nz_sum));
}

void validate_mesh_grading(const Config::MeshConfig::GradingConfig& grading) {
  if (!(grading.edge_ratio > 0.0 && grading.edge_ratio < 1.0)) {
    throw ValueError("Mesh.grid.grading.edge_ratio must be in (0, 1)");
  }
  if (grading.sg_order < 2 || (grading.sg_order % 2) != 0) {
    throw ValueError("Mesh.grid.grading.sg_order must be an even integer >= 2");
  }
  if (!(grading.sg_sigma > 0.0 && grading.sg_sigma < 1.0)) {
    throw ValueError("Mesh.grid.grading.sg_sigma must be in (0, 1)");
  }
}

void parse_mesh_grading(const py::handle grading_obj,
                        Config::MeshConfig::GradingConfig& grading) {
  if (!py::isinstance<py::dict>(grading_obj)) {
    throw_value_type_error("Mesh.grid.grading", "dict", grading_obj);
  }
  const py::dict grading_dict = py::reinterpret_borrow<py::dict>(grading_obj);
  enforce_known_keys(grading_dict, "Mesh.grid.grading",
                     {"edge_ratio", "sg_order", "sg_sigma", "mapping"});
  if (has_key(grading_dict, "edge_ratio")) {
    grading.edge_ratio =
        numeric_as_double(grading_dict["edge_ratio"], "Mesh.grid.grading.edge_ratio");
  }
  if (has_key(grading_dict, "sg_order")) {
    grading.sg_order = strict_int32(grading_dict["sg_order"], "Mesh.grid.grading.sg_order");
  }
  if (has_key(grading_dict, "sg_sigma")) {
    grading.sg_sigma =
        numeric_as_double(grading_dict["sg_sigma"], "Mesh.grid.grading.sg_sigma");
  }
  if (has_key(grading_dict, "mapping")) {
    grading.mapping =
        strict_string(grading_dict["mapping"], "Mesh.grid.grading.mapping");
    if (grading.mapping != "legacy_estimated_radius" &&
        grading.mapping != "exact_measure_v2") {
      throw ValueError(
          "Mesh.grid.grading.mapping must be \"legacy_estimated_radius\" or "
          "\"exact_measure_v2\"");
    }
  }
  validate_mesh_grading(grading);
}

bool is_motion(const std::string& value) {
  return value == "lagrangian" || value == "ale";
}

TopologyScheme parse_topology_scheme(const std::string& value,
                                      std::string_view path) {
  if (value == "single_block") {
    return TopologyScheme::SINGLE_BLOCK;
  }
  if (value == "multiblock_cart_core_polar_shell") {
    return TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL;
  }
  if (value == "multiblock_half_butterfly_5block") {
    return TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK;
  }
  if (value == "multiblock_half_butterfly_trifan_cap_5block") {
    return TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK;
  }
  if (value == "multiblock_polar_tier") {
    return TopologyScheme::MULTIBLOCK_POLAR_TIER;
  }
  if (value == "multiblock_polar_tier_cart_center") {
    return TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER;
  }
  if (value == "cone_shell_spine") {
    return TopologyScheme::CONE_SHELL_SPINE;
  }
  if (value == "pentagon_belt_shell") {
    return TopologyScheme::PENTAGON_BELT_SHELL;
  }
  throw ValueError(
      std::string(path) +
      " must be one of {\"single_block\", \"multiblock_cart_core_polar_shell\", "
      "\"multiblock_half_butterfly_5block\", "
      "\"multiblock_half_butterfly_trifan_cap_5block\", "
      "\"multiblock_polar_tier\", "
      "\"multiblock_polar_tier_cart_center\", "
      "\"cone_shell_spine\", \"pentagon_belt_shell\"}, got " +
      value);
}

MultiblockTransitionScheme parse_multiblock_transition_scheme(
    const std::string& value, std::string_view path) {
  if (value == "hermite_bridge") {
    return MultiblockTransitionScheme::HERMITE_BRIDGE;
  }
  if (value == "rounded_half_butterfly") {
    return MultiblockTransitionScheme::ROUNDED_HALF_BUTTERFLY;
  }
  if (value == "rounded_core_seam") {
    return MultiblockTransitionScheme::ROUNDED_CORE_SEAM;
  }
  throw ValueError(
      std::string(path) +
      " must be one of {\"hermite_bridge\", \"rounded_half_butterfly\", "
      "\"rounded_core_seam\"}, got " +
      value);
}

AvQcapScope parse_av_qcap_scope(const std::string& value,
                                std::string_view path) {
  if (value == "global") {
    return AvQcapScope::GLOBAL;
  }
  if (value == "tri_fan_radial_index") {
    return AvQcapScope::TRI_FAN_RADIAL_INDEX;
  }
  if (value == "centroid_r_le_r_match") {
    return AvQcapScope::CENTROID_R_LE_R_MATCH;
  }
  throw ConfigError(
      std::string(path) +
      " must be one of {\"global\", \"tri_fan_radial_index\", "
      "\"centroid_r_le_r_match\"}, got " +
      value);
}

CenterCflScope parse_center_cfl_scope(const std::string& value,
                                      std::string_view path) {
  if (value == "disabled") {
    return CenterCflScope::DISABLED;
  }
  if (value == "tri_fan_radial_index") {
    return CenterCflScope::TRI_FAN_RADIAL_INDEX;
  }
  if (value == "centroid_r_le_r_match") {
    return CenterCflScope::CENTROID_R_LE_R_MATCH;
  }
  throw ConfigError(
      std::string(path) +
      " must be one of {\"disabled\", \"tri_fan_radial_index\", "
      "\"centroid_r_le_r_match\"}, got " +
      value);
}

CenterPerturbationDiagScope parse_center_perturbation_diag_scope(
    const std::string& value,
    std::string_view path) {
  if (value == "disabled") {
    return CenterPerturbationDiagScope::DISABLED;
  }
  if (value == "tri_fan_first_ring") {
    return CenterPerturbationDiagScope::TRI_FAN_FIRST_RING;
  }
  if (value == "centroid_r_innermost_bins") {
    return CenterPerturbationDiagScope::CENTROID_R_INNERMOST_BINS;
  }
  throw ConfigError(
      std::string(path) +
      " must be one of {\"disabled\", \"tri_fan_first_ring\", "
      "\"centroid_r_innermost_bins\"}, got " +
      value);
}

bool is_mesh_tangential_target(const std::string& value) {
  return value == "lagrangian" || value == "reference";
}

bool is_state_supply_donor_mode(const std::string& value) {
  return value == "interior_per_i" || value == "interior_radial_average";
}

bool is_eos_model(const std::string& value) {
  return value == "sesame" || value == "ionmix" || value == "tmat" ||
         value == "ideal_gas" || value == "power_law_te";
}

bool is_hydro_eos_backend(const std::string& value) {
  return value == "legacy" || value == "helmholtz_spline" ||
         value == "helmholtz_jet" || value == "exact_ideal_gas" ||
         value == "rho_e_table" || value == "mie_gruneisen";
}

bool is_hydro_exact_override(const std::string& value) {
  return value == "none" || value == "pressure" || value == "sound_speed" ||
         value == "temperature" || value == "cv" || value == "temp_reclosure" ||
         value == "pressure_and_cs" || value == "p_t_cs" ||
         value == "no_writeback";
}

bool is_opacity_model(const std::string& value) {
  return value == "ionmix" || value == "sesame" || value == "constant" ||
         value == "freq_dep_marshak" || value == "table_nlte" ||
         value == "tmat" || value == "power_law" || value == "none";
}

bool is_zbar_model(const std::string& value) {
  return value == "fixed" || value == "thomas_fermi" || value == "tabular";
}

bool is_runtime_supported_opacity_model(const std::string& value) {
  return value == "constant" || value == "freq_dep_marshak" ||
         value == "table_nlte" || value == "tmat" || value == "power_law";
}

bool is_lambda_method(const std::string& value) {
  return value == "finite_difference" || value == "freeze_opacity";
}

bool is_boundary_type(const std::string& value) {
  return value == "vacuum" || value == "reflect" || value == "marshak";
}

bool is_hydro_boundary(const std::string& value) {
  return value == "free" || value == "fixed" || value == "reflect" || value == "pressure" ||
         value == "axis" || value == "state_supply";
}

bool is_ddmc_leak_stencil(const std::string& value) {
  return value == "4" || value == "9_kershaw";
}

bool is_ddmc_interface_method(const std::string& value) {
  return value == "asymptotic_diffusion_limit" || value == "marshak" ||
         value == "cleveland_gentile";
}

bool is_ddmc_interface_exit_distribution(const std::string& value) {
  return value == "cosine" || value == "half_isotropic";
}

bool is_ddmc_face_opacity_temperature(const std::string& value) {
  return value == "radiative_mean";
}

bool is_hydro_av_type(const std::string& value) {
  return value == "vnr" || value == "riemann" ||
         value == "riemann_compatible" || value == "csw";
}

bool is_csw_limiter(const std::string& value) {
  return value == "van_leer" || value == "bj";
}

bool is_ale_remap_limiter(const std::string& value) {
  return value == "van_leer" || value == "minmod";
}

bool is_plic_normal_estimator(const std::string& value) {
  return value == "youngs" || value == "LVIRA" || value == "youngs_seeded_LVIRA";
}

bool is_plic_t0_volume_cut_method(const std::string& value) {
  return value == "centroid_only_legacy" ||
         value == "adaptive_subdivision_2x2" ||
         value == "adaptive_subdivision_3x3";
}

bool is_plic_per_cell_state(const std::string& value) {
  return value == "off" || value == "sparse_on_degradation" ||
         value == "dense_debug";
}

bool is_output_format(const std::string& value) {
  return value == "hdf5";
}

bool is_output_compression(const std::string& value) {
  return value == "none" || value == "gzip";
}

bool is_production_audit_tier(const std::string& value) {
  return value == "A" || value == "B" || value == "none";
}

std::string_view block_name(const Builder::Block block) {
  switch (block) {
    case Builder::Block::Main:
      return "Main";
    case Builder::Block::Mesh:
      return "Mesh";
    case Builder::Block::Materials:
      return "Materials";
    case Builder::Block::Geometry:
      return "Geometry";
    case Builder::Block::Radiation:
      return "Radiation";
    case Builder::Block::Laser:
      return "Laser";
    case Builder::Block::Numerics:
      return "Numerics";
    case Builder::Block::Output:
      return "Output";
    case Builder::Block::Diagnostics:
      return "Diagnostics";
    case Builder::Block::Parallel:
      return "Parallel";
    case Builder::Block::Count:
      break;
  }
  return "Unknown";
}

void ensure_positive(double value, std::string_view path) {
  if (!(value > 0.0)) {
    throw ValueError(format_range_error(path, "> 0", std::to_string(value)));
  }
}

void ensure_positive_finite(double value, std::string_view path) {
  if (!(std::isfinite(value) && value > 0.0)) {
    throw ValueError(std::string(path) + " must be finite and > 0");
  }
}

using HydroBoundary2D = Config::NumericsConfig::HydroConfig::Boundary2D;

void parse_hydro_z_boundary(HydroBoundary2D::ZFaceConfig& out,
                            std::string& legacy,
                            const py::handle obj,
                            std::string_view path) {
  out = HydroBoundary2D::ZFaceConfig{};
  if (py::isinstance<py::str>(obj)) {
    out.type = normalize_boundary_mode(strict_string(obj, path), path);
    legacy = out.type;
    if (!is_hydro_boundary(out.type)) {
      throw ValueError(
          std::string(path) +
          " must be one of {\"free\", \"fixed\", \"reflect\", \"pressure\", \"axis\", \"state_supply\"}, got " +
          out.type);
    }
    if (out.type == "state_supply") {
      throw ValueError(std::string(path) +
                       "='state_supply' requires dict form with rho_g_per_cc, u_z_cm_per_s, and T_eV");
    }
    return;
  }

  if (!py::isinstance<py::dict>(obj)) {
    throw_value_type_error(path, "str|dict", obj);
  }

  const py::dict dict = py::reinterpret_borrow<py::dict>(obj);
  enforce_known_keys(dict, path,
                     {"type", "rho_g_per_cc", "u_z_cm_per_s", "T_eV",
                      "drive_t_end_s"});
  if (!has_key(dict, "type")) {
    throw ValueError(std::string(path) + ".type is required");
  }
  out.type = normalize_boundary_mode(strict_string(dict["type"], std::string(path) + ".type"),
                                     std::string(path) + ".type");
  if (!is_hydro_boundary(out.type)) {
    throw ValueError(
        std::string(path) +
        ".type must be one of {\"free\", \"fixed\", \"reflect\", \"pressure\", \"axis\", \"state_supply\"}, got " +
        out.type);
  }
  if (out.type != "state_supply") {
    throw ValueError(std::string(path) +
                     " dict form is supported only for type=\"state_supply\"");
  }
  if (!has_key(dict, "rho_g_per_cc")) {
    throw ValueError(std::string(path) + ".rho_g_per_cc is required for state_supply");
  }
  if (!has_key(dict, "u_z_cm_per_s")) {
    throw ValueError(std::string(path) + ".u_z_cm_per_s is required for state_supply");
  }
  if (!has_key(dict, "T_eV")) {
    throw ValueError(std::string(path) + ".T_eV is required for state_supply");
  }
  out.supply_rho_g_per_cc =
      numeric_as_double(dict["rho_g_per_cc"], std::string(path) + ".rho_g_per_cc");
  out.supply_u_z_cm_per_s =
      numeric_as_double(dict["u_z_cm_per_s"], std::string(path) + ".u_z_cm_per_s");
  out.supply_T_eV =
      numeric_as_double(dict["T_eV"], std::string(path) + ".T_eV");
  if (has_key(dict, "drive_t_end_s")) {
    out.drive_t_end_s = numeric_as_double(
        dict["drive_t_end_s"], std::string(path) + ".drive_t_end_s");
    if (!(std::isfinite(out.drive_t_end_s) && out.drive_t_end_s > 0.0)) {
      throw ConfigError(std::string(path) +
                        ".drive_t_end_s must be finite and > 0");
    }
  }
  if (!(out.supply_rho_g_per_cc > 0.0)) {
    throw ValueError(format_range_error(std::string(path) + ".rho_g_per_cc",
                                        "> 0",
                                        std::to_string(out.supply_rho_g_per_cc)));
  }
  if (!(out.supply_T_eV > 0.0)) {
    throw ValueError(format_range_error(std::string(path) + ".T_eV",
                                        "> 0",
                                        std::to_string(out.supply_T_eV)));
  }
  legacy = out.type;
}

void ensure_non_negative(double value, std::string_view path) {
  if (!(value >= 0.0)) {
    throw ValueError(format_range_error(path, ">= 0 (finite)", std::to_string(value)));
  }
}

void ensure_int_ge(std::int64_t value, std::int64_t lower, std::string_view path) {
  if (value < lower) {
    throw ValueError(format_range_error(path, ">= " + std::to_string(lower),
                                        std::to_string(value)));
  }
}

void validate_evacuated_cell_config(
    const Config::NumericsConfig::AleConfig::EvacuatedCellConfig& config,
    const std::string& path) {
  ensure_int_ge(config.every_n_steps, 1, path + ".every_n_steps");
  if (!(std::isfinite(config.off_mass_fraction) &&
        std::isfinite(config.arm_mass_fraction) &&
        config.off_mass_fraction > 0.0 &&
        config.off_mass_fraction < config.arm_mass_fraction &&
        config.arm_mass_fraction < 1.0)) {
    throw ValueError(
        path + " requires 0 < off_mass_fraction < arm_mass_fraction < 1");
  }
  if (!(std::isfinite(config.rho_vacuum_policy_g_per_cc) &&
        config.rho_vacuum_policy_g_per_cc > 0.0)) {
    throw ValueError(
        path + ".rho_vacuum_policy_g_per_cc must be finite and > 0");
  }
  ensure_int_ge(
      config.off_hold_evaluations, 1, path + ".off_hold_evaluations");
  if (!(std::isfinite(config.laser_ne_over_ncrit_max) &&
        config.laser_ne_over_ncrit_max > 0.0)) {
    throw ValueError(
        path + ".laser_ne_over_ncrit_max must be finite and > 0");
  }
  if (!(std::isfinite(config.laser_wavelength_nm) &&
        config.laser_wavelength_nm > 0.0)) {
    throw ValueError(path + ".laser_wavelength_nm must be finite and > 0");
  }
  if (!(std::isfinite(config.coupling_fraction_max) &&
        config.coupling_fraction_max > 0.0)) {
    throw ValueError(path + ".coupling_fraction_max must be finite and > 0");
  }
  ensure_int_ge(
      config.max_cells_per_event, 1, path + ".max_cells_per_event");
  ensure_int_ge(config.rematerialize_after_evaluations,
                1,
                path + ".rematerialize_after_evaluations");
  if (!(std::isfinite(config.rematerialize_volume_fraction) &&
        config.rematerialize_volume_fraction > 0.0 &&
        config.rematerialize_volume_fraction < 1.0)) {
    throw ValueError(
        path + ".rematerialize_volume_fraction must be finite and in (0, 1)");
  }
  if (!(std::isfinite(config.rematerialize_neighbor_change_max) &&
        config.rematerialize_neighbor_change_max > 0.0 &&
        config.rematerialize_neighbor_change_max < 1.0)) {
    throw ValueError(
        path +
        ".rematerialize_neighbor_change_max must be finite and in (0, 1)");
  }
  ensure_int_ge(config.rematerialize_dwell_evaluations,
                1,
                path + ".rematerialize_dwell_evaluations");
  const auto& closure = config.closure_contact;
  const std::string closure_path = path + ".closure_contact";
  const auto ensure_open_unit_interval = [&](const double value,
                                             const std::string& field) {
    if (!(std::isfinite(value) && value > 0.0 && value < 1.0)) {
      throw ValueError(closure_path + "." + field +
                       " must be finite and in (0, 1)");
    }
  };
  ensure_open_unit_interval(closure.gap_floor_fraction,
                            "gap_floor_fraction");
  ensure_open_unit_interval(closure.gap_arm_fraction, "gap_arm_fraction");
  ensure_open_unit_interval(closure.live_mass_gate, "live_mass_gate");
  ensure_open_unit_interval(closure.live_volume_gate, "live_volume_gate");
  ensure_open_unit_interval(closure.refill_min_mass_fraction,
                            "refill_min_mass_fraction");
  ensure_open_unit_interval(closure.refill_min_density_ratio,
                            "refill_min_density_ratio");
  ensure_open_unit_interval(closure.reengage_gap_margin,
                            "reengage_gap_margin");
  if (!(std::isfinite(closure.mortar_position_drift_beta) &&
        closure.mortar_position_drift_beta >= 0.0 &&
        closure.mortar_position_drift_beta <= 0.1)) {
    throw ValueError(closure_path +
                     ".mortar_position_drift_beta must be finite and in "
                     "[0, 0.1]");
  }
  if (!(closure.gap_arm_fraction > closure.gap_floor_fraction)) {
    throw ValueError(closure_path +
                     ".gap_arm_fraction must be greater than "
                     "gap_floor_fraction");
  }
  if (!(std::isfinite(closure.release_force_c) &&
        closure.release_force_c > 0.0)) {
    throw ValueError(closure_path +
                     ".release_force_c must be finite and > 0");
  }
  ensure_int_ge(closure.release_persistence_stages,
                1,
                closure_path + ".release_persistence_stages");
  if (closure.lcp_apply_enabled && !closure.surface_engage_enabled) {
    throw ConfigError(
        closure_path + ".lcp_apply_enabled=true requires " +
        closure_path + ".surface_engage_enabled=true");
  }
  const auto& collapse = closure.axis_edge_collapse;
  const std::string collapse_path = closure_path + ".axis_edge_collapse";
  if (collapse.persistence_window < 2 ||
      collapse.persistence_window > 8) {
    throw ValueError(collapse_path +
                     ".persistence_window must be in [2, 8]");
  }
  if (collapse.persistence_min_closing < 1 ||
      collapse.persistence_min_closing > collapse.persistence_window) {
    throw ValueError(
        collapse_path +
        ".persistence_min_closing must be in [1, persistence_window]");
  }
  const auto& strip = closure.flank_tangential_strip;
  const std::string strip_path =
      closure_path + ".flank_tangential_strip";
  if (strip.band_layers < 1 || strip.band_layers > 3) {
    throw ValueError(strip_path + ".band_layers must be in [1, 3]");
  }
  if (strip.band_halfwidth_j < 1 || strip.band_halfwidth_j > 32) {
    throw ValueError(
        strip_path + ".band_halfwidth_j must be in [1, 32]");
  }
  if (!(std::isfinite(strip.arm_quality_ratio) &&
        std::isfinite(strip.release_quality_ratio) &&
        strip.arm_quality_ratio > 0.0 &&
        strip.arm_quality_ratio < strip.release_quality_ratio &&
        strip.release_quality_ratio <= 1.0)) {
    throw ValueError(
        strip_path +
        " quality ratios must satisfy 0 < arm_quality_ratio < "
        "release_quality_ratio <= 1");
  }
  if (!(strip.min_progress_factor > 1.0)) {
    throw ConfigError(strip_path + ".min_progress_factor must be > 1.0");
  }
  if (!(std::isfinite(strip.slip_handoff_ratio) &&
        strip.slip_handoff_ratio > 0.0)) {
    throw ConfigError(strip_path + ".slip_handoff_ratio must be > 0.0");
  }
}

using ProductionAuditConfig =
    Config::NumericsConfig::DiagnosticsConfig::ProductionAuditConfig;

void parse_production_audit_regions(const py::handle obj,
                                    ProductionAuditConfig& cfg) {
  if (!py::isinstance<py::sequence>(obj) || py::isinstance<py::str>(obj)) {
    throw_value_type_error("Numerics.diagnostics.production_audit.region_of_interest",
                           "list[dict]",
                           obj);
  }
  const py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
  cfg.region_of_interest.clear();
  cfg.region_of_interest.reserve(seq.size());
  for (std::size_t i = 0; i < seq.size(); ++i) {
    const std::string base =
        "Numerics.diagnostics.production_audit.region_of_interest[" +
        std::to_string(i) + "]";
    const py::handle region_obj = seq[i];
    if (!py::isinstance<py::dict>(region_obj)) {
      throw_value_type_error(base, "dict", region_obj);
    }
    const py::dict region = py::reinterpret_borrow<py::dict>(region_obj);
    enforce_known_keys(region, base, {"i_min", "i_max", "j_min", "j_max"});
    for (const char* key : {"i_min", "i_max", "j_min", "j_max"}) {
      if (!has_key(region, key)) {
        throw ConfigError(base + " requires key '" + key + "'");
      }
    }
    ProductionAuditConfig::RegionOfInterest out;
    out.i_min = strict_int32(region["i_min"], base + ".i_min");
    out.i_max = strict_int32(region["i_max"], base + ".i_max");
    out.j_min = strict_int32(region["j_min"], base + ".j_min");
    out.j_max = strict_int32(region["j_max"], base + ".j_max");
    if (out.i_min < 0 || out.j_min < 0) {
      throw ValueError(base + " indices must be >= 0");
    }
    if (out.i_max < out.i_min || out.j_max < out.j_min) {
      throw ValueError(base + " max indices must be >= min indices");
    }
    cfg.region_of_interest.push_back(out);
  }
}

std::vector<std::string> enabled_tier_a_escape_valves(
    const Config::NumericsConfig& numerics) {
  std::vector<std::string> enabled;
  if (numerics.ale.emergency_cell_deactivation_enabled) {
    enabled.push_back("Numerics.ale.emergency_cell_deactivation_enabled");
  }
  if (numerics.ale.multi_node_boundary_repair_enabled) {
    enabled.push_back("Numerics.ale.multi_node_boundary_repair_enabled");
  }
  if (numerics.ale.multi_node_interior_repair_enabled) {
    enabled.push_back("Numerics.ale.multi_node_interior_repair_enabled");
  }
  if (numerics.ale.axis_variational_projection_enabled) {
    enabled.push_back("Numerics.ale.axis_variational_projection_enabled");
  }
  if (numerics.ale.local_boundary_repair_enabled) {
    enabled.push_back("Numerics.ale.local_boundary_repair_enabled");
  }
  if (numerics.hydro.driver_retry_active_mesh_repair_enabled) {
    enabled.push_back("Numerics.hydro.driver_retry_active_mesh_repair_enabled");
  }
  return enabled;
}

void validate_production_audit_config(const Config::NumericsConfig& numerics) {
  const auto& audit = numerics.diagnostics.production_audit;
  if (!is_production_audit_tier(audit.tier)) {
    throw ValueError(
        "Numerics.diagnostics.production_audit.tier must be one of "
        "{\"A\", \"B\", \"none\"}");
  }
  ensure_non_negative(audit.escape_valve_budget.mass_max,
                      "Numerics.diagnostics.production_audit."
                      "escape_valve_budget.mass_max");
  ensure_non_negative(audit.escape_valve_budget.energy_max,
                      "Numerics.diagnostics.production_audit."
                      "escape_valve_budget.energy_max");
  if (audit.tier == "A") {
    const std::vector<std::string> enabled =
        enabled_tier_a_escape_valves(numerics);
    if (!enabled.empty()) {
      std::ostringstream oss;
      oss << "Numerics.diagnostics.production_audit.tier=\"A\" forbids "
          << "escape-valve controls enabled at parse/validation time:";
      for (const std::string& path : enabled) {
        oss << " " << path;
      }
      throw ConfigError(oss.str());
    }
  }
}

void parse_adaptive_av_coeff(
    const py::handle obj,
    Config::NumericsConfig::HydroConfig::AdaptiveAVCoeff& coeff,
    const std::string& path) {
  if (!py::isinstance<py::dict>(obj)) {
    throw_value_type_error(path, "dict", obj);
  }
  const py::dict dict = py::reinterpret_borrow<py::dict>(obj);
  enforce_known_keys(dict, path, {"c1", "c2", "heat_C", "Cpsv", "cbulk"});
  if (has_key(dict, "c1")) {
    coeff.c1 = numeric_as_double(dict["c1"], path + ".c1");
    ensure_non_negative(coeff.c1, path + ".c1");
  }
  if (has_key(dict, "c2")) {
    coeff.c2 = numeric_as_double(dict["c2"], path + ".c2");
    ensure_non_negative(coeff.c2, path + ".c2");
  }
  if (has_key(dict, "heat_C")) {
    coeff.heat_C = numeric_as_double(dict["heat_C"], path + ".heat_C");
    ensure_non_negative(coeff.heat_C, path + ".heat_C");
  }
  if (has_key(dict, "Cpsv")) {
    coeff.Cpsv = numeric_as_double(dict["Cpsv"], path + ".Cpsv");
    ensure_non_negative(coeff.Cpsv, path + ".Cpsv");
  }
  if (has_key(dict, "cbulk")) {
    coeff.cbulk = numeric_as_double(dict["cbulk"], path + ".cbulk");
    ensure_non_negative(coeff.cbulk, path + ".cbulk");
  }
}

std::vector<double> make_log_uniform_bounds(const int n_groups,
                                            const double E_min_eV,
                                            const double E_max_eV) {
  std::vector<double> bounds;
  if (n_groups <= 0) {
    return bounds;
  }
  bounds.reserve(static_cast<std::size_t>(n_groups + 1));

  if (n_groups == 1) {
    bounds.push_back(E_min_eV);
    bounds.push_back(E_max_eV);
    return bounds;
  }

  const double log_min = std::log(E_min_eV);
  const double log_max = std::log(E_max_eV);
  for (int g = 0; g <= n_groups; ++g) {
    const double u = static_cast<double>(g) / static_cast<double>(n_groups);
    bounds.push_back(std::exp((1.0 - u) * log_min + u * log_max));
  }
  bounds.back() = E_max_eV;
  return bounds;
}

materials::IonmixZbarTable make_zero_zbar_table() {
  materials::IonmixZbarTable out;
  out.rho_grid = {1.0e-30, 1.0};
  out.T_grid_eV = {1.0e-6, 1.0};
  out.log_rho_grid = {std::log(out.rho_grid[0]), std::log(out.rho_grid[1])};
  out.log_T_grid = {std::log(out.T_grid_eV[0]), std::log(out.T_grid_eV[1])};
  out.zbar_table.assign(out.rho_grid.size() * out.T_grid_eV.size(), 0.0);
  return out;
}

double cone_shell_geometric_sum(const double ratio, const int count) {
  double sum = 0.0;
  double term = 1.0;
  for (int k = 0; k < count; ++k) {
    sum += term;
    term *= ratio;
  }
  return sum;
}

double cone_shell_strip_total(const double first_width,
                              const int layers,
                              const double growth) {
  double total = 0.0;
  double width = first_width;
  for (int layer = 0; layer < layers; ++layer) {
    total += width;
    width *= growth;
  }
  return total;
}

double cone_shell_strip_last_width(const double first_width,
                                   const int layers,
                                   const double growth) {
  double width = first_width;
  for (int layer = 1; layer < layers; ++layer) {
    width *= growth;
  }
  return width;
}

int cone_shell_mixed_ray_count(const double h_start,
                               const double distance_min,
                               const double distance_max,
                               const char* name) {
  const double count_real = std::clamp(
      std::ceil(std::log(1.0 + 0.35 * distance_max / h_start) /
                std::log(1.35)),
      6.0, 32.0);
  const int count = static_cast<int>(count_real);
  if (static_cast<double>(count) * h_start >= distance_min &&
      distance_min / static_cast<double>(count) < 0.5 * h_start) {
    std::ostringstream oss;
    oss << "cone_shell " << name
        << " mixed ladder violates the short-station chi hard band; "
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "h_start=" << h_start << ", d_min=" << distance_min
        << ", d_max=" << distance_max << ", K=" << count;
    throw ConfigError(oss.str());
  }
  if (h_start * cone_shell_geometric_sum(1.70, count) < distance_max) {
    std::ostringstream oss;
    oss << "cone_shell " << name
        << " mixed ladder requires growth above 1.7; "
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "h_start=" << h_start << ", d_max=" << distance_max
        << ", K=" << count;
    throw ConfigError(oss.str());
  }
  return count;
}

int cone_shell_tip_fill_layer_count(
    const Config::MeshConfig& mesh,
    const double h_n0) {
  const double tip_box_z = mesh.cone_shell_axis_sign > 0
                               ? mesh.box_z_min
                               : mesh.box_z_max;
  const double depth = std::abs(tip_box_z - mesh.cone_shell_tip_z);
  const double h_lt = mesh.cone_shell_tip_size_factor * h_n0;
  return static_cast<int>(
      std::clamp(std::ceil(depth / (2.0 * h_lt)), 6.0, 48.0));
}

}  // namespace

void Builder::mark_block_called(const Block block) {
  if (blocks_called.test(static_cast<std::size_t>(block))) {
    tenryu::core::log_warning(
        std::string(block_name(block)) +
        " block called multiple times; later call overwrites earlier values");
  }
  blocks_called.set(static_cast<std::size_t>(block), true);
}

void Builder::register_callable(const std::string& path,
                                const PythonCallable& callable,
                                const py::handle callable_obj) {
  callables[path] = callable;
  callable_objects[path] = py::reinterpret_borrow<py::object>(callable_obj);
}

void Builder::set_main(py::dict kwargs) {
  mark_block_called(Block::Main);
  enforce_known_keys(kwargs, "Main",
                     {"name", "dimension", "t_end", "seed", "max_steps", "verbosity",
                      "restart_from", "units", "temperature_model"});

  auto& main = config.main;
  if (has_key(kwargs, "name")) {
    main_name_explicit = true;
    main.name = strict_string(kwargs["name"], "Main.name");
  }
  if (has_key(kwargs, "dimension")) {
    main.dimension = strict_string(kwargs["dimension"], "Main.dimension");
    if (!is_dimension(main.dimension)) {
      throw ValueError("Main.dimension must be one of {\"1D_SPH\", \"1D_CYL\", "
                       "\"2D_RZ\"}, got " +
                       main.dimension);
    }
    main.dim = (main.dimension == "2D_RZ") ? 2 : 1;
  }
  if (has_key(kwargs, "t_end")) {
    main.t_end = numeric_as_double(kwargs["t_end"], "Main.t_end");
    ensure_positive(main.t_end, "Main.t_end");
  }
  if (has_key(kwargs, "seed")) {
    main.seed = strict_uint64(kwargs["seed"], "Main.seed");
  }
  if (has_key(kwargs, "max_steps")) {
    main.max_steps = strict_int32(kwargs["max_steps"], "Main.max_steps");
    ensure_int_ge(main.max_steps, 1, "Main.max_steps");
    // 2^24 - 1: Philox RNG counter-based splitting uses step in upper bits.
    // Exceeding this risks RNG stream aliasing (NUMERICS §12.7.1).
    constexpr int kMaxStepsUpperBound = 16'777'215;
    if (main.max_steps > kMaxStepsUpperBound) {
      throw ConfigError(format_range_error(
          "Main.max_steps",
          "<= " + std::to_string(kMaxStepsUpperBound),
          std::to_string(main.max_steps)));
    }
  }
  if (has_key(kwargs, "verbosity")) {
    main.verbosity = strict_string(kwargs["verbosity"], "Main.verbosity");
    if (!is_verbosity(main.verbosity)) {
      throw ValueError(
          "Main.verbosity must be one of {\"quiet\", \"normal\", \"verbose\", \"debug\"}, got " +
          main.verbosity);
    }
  }
  if (has_key(kwargs, "restart_from")) {
    main.restart_from = strict_string(kwargs["restart_from"], "Main.restart_from");
  }
  if (has_key(kwargs, "units")) {
    main.units = strict_string(kwargs["units"], "Main.units");
    if (main.units != "cgs_eV") {
      throw ConfigError("Main.units must be \"cgs_eV\" in v1.0");
    }
  }
  if (has_key(kwargs, "temperature_model")) {
    main.temperature_model =
        strict_string(kwargs["temperature_model"], "Main.temperature_model");
    if (!is_temperature_model(main.temperature_model)) {
      throw ValueError(
          "Main.temperature_model must be one of {\"1T\", \"2T\", \"auto\"}, got " +
          main.temperature_model);
    }
  }
  main.two_temperature = (main.temperature_model == "2T");
}

void Builder::set_mesh(py::dict kwargs) {
  mark_block_called(Block::Mesh);
  enforce_known_keys(kwargs, "Mesh",
                     {"nr", "nz", "r_min", "r_max", "z_min", "z_max", "grid_type_r",
                      "grid_type_z", "grid_r", "grid_z", "grid_theta", "grid", "auto_regions",
                      "zoning_intent",
                      "explicit_nodes", "explicit_nodes_z", "explicit_nodes_theta",
                      "auto_regions_axis", "auto_zone",
                      "geometry_1d", "motion",
                      "logical_mesh_2d", "polar_center_treatment",
                      "center_button_outer_node_ring",
                      "polar_equal_mu_zoning",
                      "spherical_polar_s_max", "polar_theta_min",
                      "box_r_max", "box_z_min",
                      "box_z_max", "box_center_z", "cone_shell_alpha",
                      "cone_shell_wall_thickness", "cone_shell_tip_radius",
                      "cone_shell_tip_radius_kind", "cone_shell_tip_z",
                      "cone_shell_wall_length", "cone_shell_axis_sign",
                      "cone_shell_n_cells", "cone_shell_n_growth",
                      "cone_shell_tip_size_factor",
                      "cone_shell_base_size_factor",
                      "cone_shell_tip_hold", "cone_shell_grading_length",
                      "cone_shell_l_ratio_max",
                      "cone_shell_tip_rotation_length",
                      "cone_shell_base_cut",
                      "cone_shell_base_rotation_length",
                      "cone_shell_farfield_target_measure",
                      "cone_shell_outer_vac_first_factor",
                      "cone_shell_outer_vac_layers",
                      "cone_shell_outer_vac_growth",
                      "cone_shell_inner_vac_first_factor",
                      "cone_shell_inner_vac_layers",
                      "cone_shell_inner_vac_growth",
                      "cone_shell_end_vac_first_factor",
                      "cone_shell_end_vac_layers",
                      "cone_shell_end_vac_growth", "cone_theta_wall",
                      "cone_tip_radius", "cone_activation_radius",
                      "cone_fine_cells_minus", "cone_fine_cells_plus",
                      "cone_angular_growth_max", "cone_tip_style",
                      "morph_rings", "collar_rings", "morph_growth_max",
                      "spherical_polar_kappa", "topology_scheme",
                      "pentagon_belt_layers",
                      "multiblock_cart_core_r_c",
                      "multiblock_cart_core_r_match",
                      "multiblock_cart_core_n_c",
                      "multiblock_cart_core_bridge_layers",
                      "polar_tier_cart_cut_ring",
                      "polar_tier_center_kind",
                      "multiblock_cart_core_bridge_grading",
                      "multiblock_cart_core_bridge_spacing_floor",
                      "multiblock_cart_core_bridge_ratio_max",
                      "multiblock_theta_cap_widen_factor",
                      "multiblock_transition_scheme",
                      "multiblock_cap_p",
                      "multiblock_bridge_elliptic_sweeps",
                      "multiblock_bridge_elliptic_omega",
                      "polar_tier_chi_lo", "polar_tier_chi_hi",
                      "polar_tier_belt_thickness_frac",
                      "polar_tier_belt_rows",
                      "polar_tier_pole_cap_m",
                      "polar_tier_pole_cap_alpha",
                      "polar_tier_dendrite_enabled",
                      "polar_tier_native_pentagon",
                      "shell_polar_cap_dendrite",
                      "shell_cap_rows_2x",
                      "polar_tier_dendrite_s_theta_rows_below",
                      "polar_tier_fan_sectors",
                      "polar_tier_min_tier_columns",
                      "polar_tier_fan_first_ring_radius_cm",
                      "polar_tier_hydro_enabled",
                      "multiblock_outer_svec_tangent_balance", "floors"});

  auto& mesh = config.mesh;
  const Config::MeshConfig mesh_defaults;
  mesh.grid_segments.clear();
  mesh.grid_segments_repr.clear();
  mesh.grid_segments_z.clear();
  mesh.grid_segments_z_repr.clear();
  mesh.grid_segments_theta.clear();
  mesh.grid_segments_theta_repr.clear();
  mesh.auto_regions.clear();
  mesh.zoning_intent = Config::MeshConfig::ZoningIntentNL{};
  mesh.auto_regions_axis = mesh_defaults.auto_regions_axis;
  mesh.auto_config = Config::MeshConfig::AutoZoneConfig{};
  mesh.grading = Config::MeshConfig::GradingConfig{};
  mesh.explicit_nodes.clear();
  mesh.explicit_nodes_z.clear();
  mesh.explicit_nodes_theta.clear();
  mesh.pentagon_belt_layers.clear();
  mesh.topology_scheme = TopologyScheme::SINGLE_BLOCK;
  mesh.topology_scheme_explicit = false;
  mesh.multiblock_cart_core_r_c = std::numeric_limits<double>::quiet_NaN();
  mesh.multiblock_cart_core_r_match = std::numeric_limits<double>::quiet_NaN();
  mesh.multiblock_cart_core_n_c = -1;
  mesh.multiblock_cart_core_bridge_layers = -1;
  mesh.polar_tier_cart_cut_ring = -1;
  mesh.polar_tier_center_kind = mesh_defaults.polar_tier_center_kind;
  mesh.multiblock_cart_core_bridge_grading =
      mesh_defaults.multiblock_cart_core_bridge_grading;
  mesh.multiblock_cart_core_bridge_spacing_floor =
      mesh_defaults.multiblock_cart_core_bridge_spacing_floor;
  mesh.multiblock_cart_core_bridge_ratio_max =
      mesh_defaults.multiblock_cart_core_bridge_ratio_max;
  mesh.multiblock_theta_cap_widen_factor =
      mesh_defaults.multiblock_theta_cap_widen_factor;
  mesh.multiblock_transition_scheme =
      mesh_defaults.multiblock_transition_scheme;
  mesh.multiblock_cap_p = mesh_defaults.multiblock_cap_p;
  mesh.multiblock_bridge_elliptic_sweeps =
      mesh_defaults.multiblock_bridge_elliptic_sweeps;
  mesh.multiblock_bridge_elliptic_omega =
      mesh_defaults.multiblock_bridge_elliptic_omega;
  mesh.polar_tier_chi_lo = mesh_defaults.polar_tier_chi_lo;
  mesh.polar_tier_chi_hi = mesh_defaults.polar_tier_chi_hi;
  mesh.polar_tier_belt_thickness_frac =
      mesh_defaults.polar_tier_belt_thickness_frac;
  mesh.polar_tier_belt_rows = mesh_defaults.polar_tier_belt_rows;
  mesh.polar_tier_pole_cap_m = mesh_defaults.polar_tier_pole_cap_m;
  mesh.polar_tier_pole_cap_alpha = mesh_defaults.polar_tier_pole_cap_alpha;
  mesh.polar_tier_dendrite_enabled =
      mesh_defaults.polar_tier_dendrite_enabled;
  mesh.polar_tier_native_pentagon =
      mesh_defaults.polar_tier_native_pentagon;
  mesh.shell_polar_cap_dendrite =
      mesh_defaults.shell_polar_cap_dendrite;
  mesh.shell_cap_rows_2x = mesh_defaults.shell_cap_rows_2x;
  mesh.polar_tier_dendrite_s_theta_rows_below =
      mesh_defaults.polar_tier_dendrite_s_theta_rows_below;
  mesh.polar_tier_fan_sectors = mesh_defaults.polar_tier_fan_sectors;
  mesh.polar_tier_min_tier_columns =
      mesh_defaults.polar_tier_min_tier_columns;
  mesh.polar_tier_fan_first_ring_radius_cm =
      mesh_defaults.polar_tier_fan_first_ring_radius_cm;
  mesh.polar_tier_hydro_enabled =
      mesh_defaults.polar_tier_hydro_enabled;
  mesh.multiblock_outer_svec_tangent_balance =
      mesh_defaults.multiblock_outer_svec_tangent_balance;
  bool grading_specified = false;
  if (has_key(kwargs, "topology_scheme")) {
    mesh.topology_scheme_explicit = true;
    mesh.topology_scheme = parse_topology_scheme(
        strict_string(kwargs["topology_scheme"], "Mesh.topology_scheme"),
        "Mesh.topology_scheme");
  }
  if (has_key(kwargs, "nr")) {
    mesh.nr = strict_int32(kwargs["nr"], "Mesh.nr");
    ensure_int_ge(mesh.nr, 4, "Mesh.nr");
  }
  if (has_key(kwargs, "nz")) {
    mesh.nz = strict_int32(kwargs["nz"], "Mesh.nz");
    ensure_int_ge(mesh.nz, 1, "Mesh.nz");
  }
  if (has_key(kwargs, "r_min")) {
    mesh_r_min_explicit = true;
    mesh.r_min = numeric_as_double(kwargs["r_min"], "Mesh.r_min");
  }
  if (has_key(kwargs, "r_max")) {
    mesh_r_max_explicit = true;
    mesh.r_max = numeric_as_double(kwargs["r_max"], "Mesh.r_max");
  }
  if (has_key(kwargs, "z_min")) {
    mesh_z_min_explicit = true;
    mesh.z_min = numeric_as_double(kwargs["z_min"], "Mesh.z_min");
  }
  if (has_key(kwargs, "z_max")) {
    mesh_z_max_explicit = true;
    mesh.z_max = numeric_as_double(kwargs["z_max"], "Mesh.z_max");
  }
  if (has_key(kwargs, "geometry_1d")) {
    config.mesh.geometry_1d =
        strict_string(kwargs["geometry_1d"], "Mesh.geometry_1d");
    if (config.mesh.geometry_1d != "spherical" &&
        config.mesh.geometry_1d != "cylindrical" &&
        config.mesh.geometry_1d != "planar") {
      throw ConfigError(
          "Mesh.geometry_1d must be \"spherical\", \"cylindrical\", or"
          " \"planar\"");
    }
  }
  if (has_key(kwargs, "grid")) {
    const py::handle grid_obj = kwargs["grid"];
    std::string grid_type;
    if (py::isinstance<py::str>(grid_obj)) {
      grid_type = strict_string(grid_obj, "Mesh.grid");
    } else if (py::isinstance<py::dict>(grid_obj)) {
      const py::dict grid = py::reinterpret_borrow<py::dict>(grid_obj);
      if (!has_key(grid, "type")) {
        throw ConfigError("Mesh.grid dict requires key 'type'");
      }
      grid_type = strict_string(grid["type"], "Mesh.grid.type");
      if (grid_type == "graded") {
        enforce_known_keys(grid, "Mesh.grid", {"type", "segments", "grading"});
        if (!has_key(grid, "segments")) {
          throw ConfigError("Mesh.grid.type='graded' requires Mesh.grid.segments");
        }
        parse_mesh_grid_segments(grid["segments"], mesh);
        if (has_key(grid, "grading")) {
          if (grading_specified) {
            throw ConfigError(
                "Mesh grading may be specified in only one grid block");
          }
          grading_specified = true;
          parse_mesh_grading(grid["grading"], mesh.grading);
        } else {
          validate_mesh_grading(mesh.grading);
        }
      } else if (grid_type == "uniform") {
        enforce_known_keys(grid, "Mesh.grid", {"type"});
      } else {
        throw ValueError("Mesh.grid.type must be one of {\"uniform\", \"graded\"}, got " +
                         grid_type);
      }
    } else {
      throw_value_type_error("Mesh.grid", "str|dict", grid_obj);
    }
    if (!is_grid_type(grid_type)) {
      throw ValueError("Mesh.grid must be one of {\"uniform\", \"graded\"}, got " +
                       grid_type);
    }
    mesh.grid_type_r = grid_type;
    mesh.grid_type_z = grid_type;
  }
  if (has_key(kwargs, "grid_type_r")) {
    mesh.grid_type_r = strict_string(kwargs["grid_type_r"], "Mesh.grid_type_r");
    if (!is_grid_type(mesh.grid_type_r)) {
      throw ValueError("Mesh.grid_type_r must be one of {\"uniform\", \"graded\"}, got " +
                       mesh.grid_type_r);
    }
  }
  if (has_key(kwargs, "grid_r")) {
    const py::handle grid_r_obj = kwargs["grid_r"];
    if (py::isinstance<py::str>(grid_r_obj)) {
      if (has_key(kwargs, "grid_type_r")) {
        tenryu::core::log_warning(
            "Mesh.grid_r is ignored because Mesh.grid_type_r is also provided");
      } else {
        mesh.grid_type_r = strict_string(grid_r_obj, "Mesh.grid_r");
        if (!is_grid_type(mesh.grid_type_r)) {
          throw ValueError(
              "Mesh.grid_r must be one of {\"uniform\", \"graded\"}, got " +
              mesh.grid_type_r);
        }
      }
    } else if (py::isinstance<py::dict>(grid_r_obj)) {
      if (has_key(kwargs, "grid_type_r")) {
        throw ConfigError(
            "Mesh.grid_r dict is ambiguous with Mesh.grid_type_r");
      }
      const py::dict grid_r = py::reinterpret_borrow<py::dict>(grid_r_obj);
      if (!has_key(grid_r, "type")) {
        throw ConfigError("Mesh.grid_r dict requires key 'type'");
      }
      const std::string grid_type_r =
          strict_string(grid_r["type"], "Mesh.grid_r.type");
      if (grid_type_r == "graded") {
        enforce_known_keys(grid_r, "Mesh.grid_r",
                           {"type", "segments", "grading"});
        if (!has_key(grid_r, "segments")) {
          throw ConfigError(
              "Mesh.grid_r.type='graded' requires Mesh.grid_r.segments");
        }
        parse_mesh_grid_segments(grid_r["segments"], mesh);
        if (has_key(grid_r, "grading")) {
          if (grading_specified) {
            throw ConfigError(
                "Mesh grading may be specified in only one grid block");
          }
          grading_specified = true;
          parse_mesh_grading(grid_r["grading"], mesh.grading);
        } else {
          validate_mesh_grading(mesh.grading);
        }
      } else if (grid_type_r == "uniform") {
        enforce_known_keys(grid_r, "Mesh.grid_r", {"type"});
      } else {
        throw ValueError(
            "Mesh.grid_r.type must be one of {\"uniform\", \"graded\"}, got " +
            grid_type_r);
      }
      mesh.grid_type_r = grid_type_r;
    } else {
      throw_value_type_error("Mesh.grid_r", "str|dict", grid_r_obj);
    }
  }
  if (has_key(kwargs, "grid_type_z")) {
    mesh.grid_type_z = strict_string(kwargs["grid_type_z"], "Mesh.grid_type_z");
    if (!is_grid_type(mesh.grid_type_z)) {
      throw ValueError("Mesh.grid_type_z must be one of {\"uniform\", \"graded\"}, got " +
                       mesh.grid_type_z);
    }
  }
  if (has_key(kwargs, "grid_z")) {
    const py::handle grid_z_obj = kwargs["grid_z"];
    if (py::isinstance<py::str>(grid_z_obj)) {
      if (has_key(kwargs, "grid_type_z")) {
        tenryu::core::log_warning(
            "Mesh.grid_z is ignored because Mesh.grid_type_z is also provided");
      } else {
        mesh.grid_type_z = strict_string(grid_z_obj, "Mesh.grid_z");
        if (!is_grid_type(mesh.grid_type_z)) {
          throw ValueError(
              "Mesh.grid_z must be one of {\"uniform\", \"graded\"}, got " +
              mesh.grid_type_z);
        }
      }
    } else if (py::isinstance<py::dict>(grid_z_obj)) {
      if (has_key(kwargs, "grid_type_z")) {
        throw ConfigError(
            "Mesh.grid_z dict is ambiguous with Mesh.grid_type_z");
      }
      const py::dict grid_z = py::reinterpret_borrow<py::dict>(grid_z_obj);
      if (!has_key(grid_z, "type")) {
        throw ConfigError("Mesh.grid_z dict requires key 'type'");
      }
      const std::string grid_type_z =
          strict_string(grid_z["type"], "Mesh.grid_z.type");
      if (grid_type_z == "graded") {
        enforce_known_keys(grid_z, "Mesh.grid_z",
                           {"type", "segments", "grading"});
        if (!has_key(grid_z, "segments")) {
          throw ConfigError(
              "Mesh.grid_z.type='graded' requires Mesh.grid_z.segments");
        }
        parse_mesh_grid_segments_z(grid_z["segments"], mesh);
        if (has_key(grid_z, "grading")) {
          if (grading_specified) {
            throw ConfigError(
                "Mesh grading may be specified in only one grid block");
          }
          grading_specified = true;
          parse_mesh_grading(grid_z["grading"], mesh.grading);
        } else {
          validate_mesh_grading(mesh.grading);
        }
      } else if (grid_type_z == "uniform") {
        enforce_known_keys(grid_z, "Mesh.grid_z", {"type"});
      } else {
        throw ValueError(
            "Mesh.grid_z.type must be one of {\"uniform\", \"graded\"}, got " +
            grid_type_z);
      }
      mesh.grid_type_z = grid_type_z;
    } else {
      throw_value_type_error("Mesh.grid_z", "str|dict", grid_z_obj);
    }
  }
  if (has_key(kwargs, "grid_theta")) {
    const py::handle grid_theta_obj = kwargs["grid_theta"];
    if (!py::isinstance<py::dict>(grid_theta_obj)) {
      throw ConfigError(
          "Mesh.grid_theta must be a dict(type='graded', segments=[...])");
    }
    const py::dict grid_theta =
        py::reinterpret_borrow<py::dict>(grid_theta_obj);
    enforce_known_keys(grid_theta, "Mesh.grid_theta",
                       {"type", "segments", "grading"});
    if (!has_key(grid_theta, "type")) {
      throw ConfigError("Mesh.grid_theta dict requires key 'type'");
    }
    const std::string grid_type_theta =
        strict_string(grid_theta["type"], "Mesh.grid_theta.type");
    if (grid_type_theta != "graded") {
      throw ConfigError("Mesh.grid_theta.type must be 'graded'");
    }
    if (!has_key(grid_theta, "segments")) {
      throw ConfigError(
          "Mesh.grid_theta.type='graded' requires Mesh.grid_theta.segments");
    }
    parse_mesh_grid_segments_into(
        grid_theta["segments"], "Mesh.grid_theta.segments",
        mesh.grid_segments_theta, mesh.grid_segments_theta_repr);
    if (has_key(grid_theta, "grading")) {
      if (grading_specified) {
        throw ConfigError(
            "Mesh grading may be specified in only one grid block");
      }
      grading_specified = true;
      parse_mesh_grading(grid_theta["grading"], mesh.grading);
    } else {
      validate_mesh_grading(mesh.grading);
    }
  }
  if (has_key(kwargs, "motion")) {
    mesh.motion = strict_string(kwargs["motion"], "Mesh.motion");
    if (!is_motion(mesh.motion)) {
      throw ValueError(
          "Mesh.motion must be one of {\"lagrangian\", \"ale\"}, got " + mesh.motion);
    }
    motion_explicitly_set = true;
  }
  if (has_key(kwargs, "logical_mesh_2d")) {
    mesh.logical_mesh_2d = strict_string(kwargs["logical_mesh_2d"], "Mesh.logical_mesh_2d");
    if (mesh.logical_mesh_2d != "rectangular_rz" &&
        mesh.logical_mesh_2d != "spherical_polar_halfplane" &&
        mesh.logical_mesh_2d != "polar_in_box" &&
        mesh.logical_mesh_2d != "cone_shell") {
      throw ValueError(
          "Mesh.logical_mesh_2d must be one of {\"rectangular_rz\", "
          "\"spherical_polar_halfplane\", \"polar_in_box\", "
          "\"cone_shell\"}");
    }
  }
  if (has_key(kwargs, "polar_center_treatment")) {
    mesh.polar_center_treatment =
        strict_string(kwargs["polar_center_treatment"], "Mesh.polar_center_treatment");
    if (mesh.polar_center_treatment != "annular" &&
        mesh.polar_center_treatment != "tri_fan" &&
        mesh.polar_center_treatment != "button") {
      throw ValueError(
          "Mesh.polar_center_treatment must be one of {\"annular\", \"tri_fan\", \"button\"}");
    }
  }
  if (has_key(kwargs, "center_button_outer_node_ring")) {
    mesh.center_button_outer_node_ring =
        strict_int32(kwargs["center_button_outer_node_ring"],
                     "Mesh.center_button_outer_node_ring");
  }
  if (has_key(kwargs, "polar_equal_mu_zoning")) {
    mesh.polar_equal_mu_zoning =
        strict_bool(kwargs["polar_equal_mu_zoning"], "Mesh.polar_equal_mu_zoning");
  }
  if (has_key(kwargs, "spherical_polar_s_max")) {
    mesh.spherical_polar_s_max =
        numeric_as_double(kwargs["spherical_polar_s_max"], "Mesh.spherical_polar_s_max");
  }
  if (has_key(kwargs, "polar_theta_min")) {
    mesh.polar_theta_min =
        numeric_as_double(kwargs["polar_theta_min"], "Mesh.polar_theta_min");
  }
  if (has_key(kwargs, "box_r_max")) {
    mesh.box_r_max =
        numeric_as_double(kwargs["box_r_max"], "Mesh.box_r_max");
  }
  if (has_key(kwargs, "box_z_min")) {
    mesh.box_z_min =
        numeric_as_double(kwargs["box_z_min"], "Mesh.box_z_min");
  }
  if (has_key(kwargs, "box_z_max")) {
    mesh.box_z_max =
        numeric_as_double(kwargs["box_z_max"], "Mesh.box_z_max");
  }
  if (has_key(kwargs, "box_center_z")) {
    mesh.box_center_z =
        numeric_as_double(kwargs["box_center_z"], "Mesh.box_center_z");
  }
  if (has_key(kwargs, "cone_shell_alpha")) {
    mesh.cone_shell_alpha = numeric_as_double(
        kwargs["cone_shell_alpha"], "Mesh.cone_shell_alpha");
  }
  if (has_key(kwargs, "cone_shell_wall_thickness")) {
    mesh.cone_shell_wall_thickness = numeric_as_double(
        kwargs["cone_shell_wall_thickness"],
        "Mesh.cone_shell_wall_thickness");
  }
  if (has_key(kwargs, "cone_shell_tip_radius")) {
    mesh.cone_shell_tip_radius = numeric_as_double(
        kwargs["cone_shell_tip_radius"], "Mesh.cone_shell_tip_radius");
  }
  if (has_key(kwargs, "cone_shell_tip_radius_kind")) {
    mesh.cone_shell_tip_radius_kind = strict_string(
        kwargs["cone_shell_tip_radius_kind"],
        "Mesh.cone_shell_tip_radius_kind");
  }
  if (has_key(kwargs, "cone_shell_tip_z")) {
    mesh.cone_shell_tip_z = numeric_as_double(
        kwargs["cone_shell_tip_z"], "Mesh.cone_shell_tip_z");
  }
  if (has_key(kwargs, "cone_shell_wall_length")) {
    mesh.cone_shell_wall_length = numeric_as_double(
        kwargs["cone_shell_wall_length"], "Mesh.cone_shell_wall_length");
  }
  if (has_key(kwargs, "cone_shell_axis_sign")) {
    mesh.cone_shell_axis_sign = strict_int32(
        kwargs["cone_shell_axis_sign"], "Mesh.cone_shell_axis_sign");
  }
  if (has_key(kwargs, "cone_shell_n_cells")) {
    mesh.cone_shell_n_cells = strict_int32(
        kwargs["cone_shell_n_cells"], "Mesh.cone_shell_n_cells");
  }
  if (has_key(kwargs, "cone_shell_n_growth")) {
    mesh.cone_shell_n_growth = numeric_as_double(
        kwargs["cone_shell_n_growth"], "Mesh.cone_shell_n_growth");
  }
  if (has_key(kwargs, "cone_shell_tip_size_factor")) {
    mesh.cone_shell_tip_size_factor = numeric_as_double(
        kwargs["cone_shell_tip_size_factor"],
        "Mesh.cone_shell_tip_size_factor");
  }
  if (has_key(kwargs, "cone_shell_base_size_factor")) {
    mesh.cone_shell_base_size_factor = numeric_as_double(
        kwargs["cone_shell_base_size_factor"],
        "Mesh.cone_shell_base_size_factor");
  }
  if (has_key(kwargs, "cone_shell_tip_hold")) {
    mesh.cone_shell_tip_hold = numeric_as_double(
        kwargs["cone_shell_tip_hold"], "Mesh.cone_shell_tip_hold");
  }
  if (has_key(kwargs, "cone_shell_grading_length")) {
    mesh.cone_shell_grading_length = numeric_as_double(
        kwargs["cone_shell_grading_length"],
        "Mesh.cone_shell_grading_length");
  }
  if (has_key(kwargs, "cone_shell_l_ratio_max")) {
    mesh.cone_shell_l_ratio_max = numeric_as_double(
        kwargs["cone_shell_l_ratio_max"], "Mesh.cone_shell_l_ratio_max");
  }
  if (has_key(kwargs, "cone_shell_tip_rotation_length")) {
    mesh.cone_shell_tip_rotation_length = numeric_as_double(
        kwargs["cone_shell_tip_rotation_length"],
        "Mesh.cone_shell_tip_rotation_length");
  }
  if (has_key(kwargs, "cone_shell_base_cut")) {
    mesh.cone_shell_base_cut = strict_string(
        kwargs["cone_shell_base_cut"], "Mesh.cone_shell_base_cut");
  }
  if (has_key(kwargs, "cone_shell_base_rotation_length")) {
    mesh.cone_shell_base_rotation_length = numeric_as_double(
        kwargs["cone_shell_base_rotation_length"],
        "Mesh.cone_shell_base_rotation_length");
  }
  if (has_key(kwargs, "cone_shell_farfield_target_measure")) {
    mesh.cone_shell_farfield_target_measure = strict_string(
        kwargs["cone_shell_farfield_target_measure"],
        "Mesh.cone_shell_farfield_target_measure");
  }
  if (has_key(kwargs, "cone_shell_outer_vac_first_factor")) {
    mesh.cone_shell_outer_vac_first_factor = numeric_as_double(
        kwargs["cone_shell_outer_vac_first_factor"],
        "Mesh.cone_shell_outer_vac_first_factor");
  }
  if (has_key(kwargs, "cone_shell_outer_vac_layers")) {
    mesh.cone_shell_outer_vac_layers = strict_int32(
        kwargs["cone_shell_outer_vac_layers"],
        "Mesh.cone_shell_outer_vac_layers");
  }
  if (has_key(kwargs, "cone_shell_outer_vac_growth")) {
    mesh.cone_shell_outer_vac_growth = numeric_as_double(
        kwargs["cone_shell_outer_vac_growth"],
        "Mesh.cone_shell_outer_vac_growth");
  }
  if (has_key(kwargs, "cone_shell_inner_vac_first_factor")) {
    mesh.cone_shell_inner_vac_first_factor = numeric_as_double(
        kwargs["cone_shell_inner_vac_first_factor"],
        "Mesh.cone_shell_inner_vac_first_factor");
  }
  if (has_key(kwargs, "cone_shell_inner_vac_layers")) {
    mesh.cone_shell_inner_vac_layers = strict_int32(
        kwargs["cone_shell_inner_vac_layers"],
        "Mesh.cone_shell_inner_vac_layers");
  }
  if (has_key(kwargs, "cone_shell_inner_vac_growth")) {
    mesh.cone_shell_inner_vac_growth = numeric_as_double(
        kwargs["cone_shell_inner_vac_growth"],
        "Mesh.cone_shell_inner_vac_growth");
  }
  if (has_key(kwargs, "cone_shell_end_vac_first_factor")) {
    mesh.cone_shell_end_vac_first_factor = numeric_as_double(
        kwargs["cone_shell_end_vac_first_factor"],
        "Mesh.cone_shell_end_vac_first_factor");
  }
  if (has_key(kwargs, "cone_shell_end_vac_layers")) {
    mesh.cone_shell_end_vac_layers = strict_int32(
        kwargs["cone_shell_end_vac_layers"],
        "Mesh.cone_shell_end_vac_layers");
  }
  if (has_key(kwargs, "cone_shell_end_vac_growth")) {
    mesh.cone_shell_end_vac_growth = numeric_as_double(
        kwargs["cone_shell_end_vac_growth"],
        "Mesh.cone_shell_end_vac_growth");
  }
  if (has_key(kwargs, "cone_theta_wall")) {
    mesh.cone_theta_wall = numeric_as_double(
        kwargs["cone_theta_wall"], "Mesh.cone_theta_wall");
  }
  if (has_key(kwargs, "cone_tip_radius")) {
    mesh.cone_tip_radius = numeric_as_double(
        kwargs["cone_tip_radius"], "Mesh.cone_tip_radius");
  }
  if (has_key(kwargs, "cone_activation_radius")) {
    mesh.cone_activation_radius = numeric_as_double(
        kwargs["cone_activation_radius"], "Mesh.cone_activation_radius");
  }
  if (has_key(kwargs, "cone_fine_cells_minus")) {
    mesh.cone_fine_cells_minus = strict_int32(
        kwargs["cone_fine_cells_minus"], "Mesh.cone_fine_cells_minus");
  }
  if (has_key(kwargs, "cone_fine_cells_plus")) {
    mesh.cone_fine_cells_plus = strict_int32(
        kwargs["cone_fine_cells_plus"], "Mesh.cone_fine_cells_plus");
  }
  if (has_key(kwargs, "cone_angular_growth_max")) {
    mesh.cone_angular_growth_max = numeric_as_double(
        kwargs["cone_angular_growth_max"],
        "Mesh.cone_angular_growth_max");
  }
  if (has_key(kwargs, "cone_tip_style")) {
    mesh.cone_tip_style =
        strict_string(kwargs["cone_tip_style"], "Mesh.cone_tip_style");
  }
  if (has_key(kwargs, "morph_rings")) {
    mesh.morph_rings = strict_int32(kwargs["morph_rings"], "Mesh.morph_rings");
  }
  if (has_key(kwargs, "collar_rings")) {
    mesh.collar_rings =
        strict_int32(kwargs["collar_rings"], "Mesh.collar_rings");
  }
  if (has_key(kwargs, "morph_growth_max")) {
    mesh.morph_growth_max = numeric_as_double(
        kwargs["morph_growth_max"], "Mesh.morph_growth_max");
  }
  if (has_key(kwargs, "spherical_polar_kappa")) {
    mesh.spherical_polar_kappa =
        numeric_as_double(kwargs["spherical_polar_kappa"], "Mesh.spherical_polar_kappa");
  }
  if (has_key(kwargs, "pentagon_belt_layers")) {
    mesh.pentagon_belt_layers = strict_int_vector(
        kwargs["pentagon_belt_layers"], "Mesh.pentagon_belt_layers");
  }
  const bool has_multiblock_key =
      has_key(kwargs, "multiblock_cart_core_r_c") ||
      has_key(kwargs, "multiblock_cart_core_r_match") ||
      has_key(kwargs, "multiblock_cart_core_n_c") ||
      has_key(kwargs, "multiblock_cart_core_bridge_layers") ||
      has_key(kwargs, "multiblock_cart_core_bridge_grading") ||
      has_key(kwargs, "multiblock_cart_core_bridge_spacing_floor") ||
      has_key(kwargs, "multiblock_cart_core_bridge_ratio_max") ||
      has_key(kwargs, "multiblock_transition_scheme") ||
      has_key(kwargs, "multiblock_cap_p") ||
      has_key(kwargs, "multiblock_bridge_elliptic_sweeps") ||
      has_key(kwargs, "multiblock_bridge_elliptic_omega") ||
      has_key(kwargs, "polar_tier_chi_lo") ||
      has_key(kwargs, "polar_tier_chi_hi") ||
      has_key(kwargs, "polar_tier_belt_thickness_frac") ||
      has_key(kwargs, "polar_tier_belt_rows") ||
      has_key(kwargs, "polar_tier_pole_cap_m") ||
      has_key(kwargs, "polar_tier_pole_cap_alpha") ||
      has_key(kwargs, "polar_tier_dendrite_enabled") ||
      has_key(kwargs, "polar_tier_native_pentagon") ||
      has_key(kwargs, "shell_polar_cap_dendrite") ||
      has_key(kwargs, "shell_cap_rows_2x") ||
      has_key(kwargs, "polar_tier_dendrite_s_theta_rows_below") ||
      has_key(kwargs, "polar_tier_fan_sectors") ||
      has_key(kwargs, "polar_tier_min_tier_columns") ||
      has_key(kwargs, "polar_tier_fan_first_ring_radius_cm") ||
      has_key(kwargs, "polar_tier_hydro_enabled") ||
      has_key(kwargs, "multiblock_outer_svec_tangent_balance");
  const bool has_polar_tier_key =
      has_key(kwargs, "polar_tier_chi_lo") ||
      has_key(kwargs, "polar_tier_chi_hi") ||
      has_key(kwargs, "polar_tier_belt_thickness_frac") ||
      has_key(kwargs, "polar_tier_belt_rows") ||
      has_key(kwargs, "polar_tier_pole_cap_m") ||
      has_key(kwargs, "polar_tier_pole_cap_alpha") ||
      has_key(kwargs, "polar_tier_dendrite_enabled") ||
      has_key(kwargs, "polar_tier_native_pentagon") ||
      has_key(kwargs, "shell_polar_cap_dendrite") ||
      has_key(kwargs, "shell_cap_rows_2x") ||
      has_key(kwargs, "polar_tier_dendrite_s_theta_rows_below") ||
      has_key(kwargs, "polar_tier_fan_sectors") ||
      has_key(kwargs, "polar_tier_min_tier_columns") ||
      has_key(kwargs, "polar_tier_fan_first_ring_radius_cm") ||
      has_key(kwargs, "polar_tier_hydro_enabled");
  const bool has_cart_center_key =
      has_key(kwargs, "multiblock_cart_core_r_c") ||
      has_key(kwargs, "multiblock_cart_core_n_c") ||
      has_key(kwargs, "multiblock_cart_core_bridge_layers") ||
      has_key(kwargs, "multiblock_cart_core_bridge_grading") ||
      has_key(kwargs, "multiblock_cart_core_bridge_spacing_floor") ||
      has_key(kwargs, "multiblock_cart_core_bridge_ratio_max") ||
      has_key(kwargs, "center_button_outer_node_ring") ||
      has_key(kwargs, "multiblock_theta_cap_widen_factor") ||
      has_key(kwargs, "multiblock_transition_scheme") ||
      has_key(kwargs, "multiblock_cap_p") ||
      has_key(kwargs, "multiblock_bridge_elliptic_sweeps") ||
      has_key(kwargs, "multiblock_bridge_elliptic_omega") ||
      has_key(kwargs, "multiblock_outer_svec_tangent_balance");
  if (has_polar_tier_key &&
      mesh.topology_scheme != TopologyScheme::MULTIBLOCK_POLAR_TIER &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER) {
    throw ConfigError(
        "Mesh.polar_tier_* keys require "
        "Mesh.topology_scheme=\"multiblock_polar_tier\" or "
        "\"multiblock_polar_tier_cart_center\"");
  }
  if (mesh.topology_scheme == TopologyScheme::MULTIBLOCK_POLAR_TIER &&
      has_cart_center_key) {
    throw ConfigError(
        "Mesh.topology_scheme=\"multiblock_polar_tier\" rejects "
        "cart-core/button/tri-fan-specific multiblock keys");
  }
  if (has_key(kwargs, "polar_tier_cart_cut_ring")) {
    mesh.polar_tier_cart_cut_ring =
        strict_int32(kwargs["polar_tier_cart_cut_ring"],
                     "Mesh.polar_tier_cart_cut_ring");
  }
  if (mesh.polar_tier_cart_cut_ring != -1 &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER) {
    throw ConfigError(
        "Mesh.polar_tier_cart_cut_ring requires "
        "Mesh.topology_scheme=\"multiblock_polar_tier_cart_center\"");
  }
  if (has_key(kwargs, "polar_tier_center_kind")) {
    mesh.polar_tier_center_kind = strict_string(
        kwargs["polar_tier_center_kind"],
        "Mesh.polar_tier_center_kind");
  }
  if (mesh.polar_tier_center_kind != "cart_box" &&
      mesh.polar_tier_center_kind != "trifan_cap") {
    throw ValueError(
        "Mesh.polar_tier_center_kind must be one of "
        "{\"cart_box\", \"trifan_cap\"}");
  }
  if (mesh.polar_tier_center_kind != "cart_box" &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER) {
    throw ConfigError(
        "Mesh.polar_tier_center_kind is only meaningful for "
        "Mesh.topology_scheme=\"multiblock_polar_tier_cart_center\"");
  }
  if (has_key(kwargs, "multiblock_cart_core_bridge_grading") &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER) {
    throw ConfigError(
        "multiblock_cart_core_bridge_grading requires "
        "multiblock_cart_core_polar_shell or "
        "multiblock_polar_tier_cart_center topology");
  }
  if (has_key(kwargs, "multiblock_cart_core_bridge_spacing_floor") &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER) {
    throw ConfigError(
        "multiblock_cart_core_bridge_spacing_floor requires "
        "multiblock_polar_tier_cart_center topology");
  }
  if (has_key(kwargs, "multiblock_cart_core_bridge_ratio_max") &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER) {
    throw ConfigError(
        "multiblock_cart_core_bridge_ratio_max requires "
        "multiblock_polar_tier_cart_center topology");
  }
  if (mesh.topology_scheme == TopologyScheme::SINGLE_BLOCK && has_multiblock_key) {
    throw ConfigError(
        "Mesh.multiblock_* requires "
        "Mesh.topology_scheme=\"multiblock_cart_core_polar_shell\" or "
        "\"multiblock_half_butterfly_5block\" or "
        "\"multiblock_half_butterfly_trifan_cap_5block\", "
        "\"pentagon_belt_shell\" or "
        "\"multiblock_polar_tier\" or "
        "\"multiblock_polar_tier_cart_center\"");
  }
  if (has_key(kwargs, "multiblock_cart_core_r_c")) {
    mesh.multiblock_cart_core_r_c =
        numeric_as_double(kwargs["multiblock_cart_core_r_c"],
                          "Mesh.multiblock_cart_core_r_c");
  }
  if (has_key(kwargs, "multiblock_cart_core_r_match")) {
    mesh.multiblock_cart_core_r_match =
        numeric_as_double(kwargs["multiblock_cart_core_r_match"],
                          "Mesh.multiblock_cart_core_r_match");
  }
  if (has_key(kwargs, "multiblock_cart_core_n_c")) {
    mesh.multiblock_cart_core_n_c =
        strict_int32(kwargs["multiblock_cart_core_n_c"],
                     "Mesh.multiblock_cart_core_n_c");
  }
  if (has_key(kwargs, "multiblock_cart_core_bridge_layers")) {
    mesh.multiblock_cart_core_bridge_layers =
        strict_int32(kwargs["multiblock_cart_core_bridge_layers"],
                     "Mesh.multiblock_cart_core_bridge_layers");
  }
  if (has_key(kwargs, "multiblock_cart_core_bridge_grading")) {
    mesh.multiblock_cart_core_bridge_grading = strict_string(
        kwargs["multiblock_cart_core_bridge_grading"],
        "Mesh.multiblock_cart_core_bridge_grading");
    if (mesh.multiblock_cart_core_bridge_grading != "uniform" &&
        mesh.multiblock_cart_core_bridge_grading != "quintic_log" &&
        mesh.multiblock_cart_core_bridge_grading != "log") {
      throw ConfigError(
          "Mesh.multiblock_cart_core_bridge_grading must be one of "
          "{\"uniform\", \"quintic_log\", \"log\"}, got " +
          mesh.multiblock_cart_core_bridge_grading);
    }
  }
  if (has_key(kwargs, "multiblock_cart_core_bridge_spacing_floor")) {
    mesh.multiblock_cart_core_bridge_spacing_floor = numeric_as_double(
        kwargs["multiblock_cart_core_bridge_spacing_floor"],
        "Mesh.multiblock_cart_core_bridge_spacing_floor");
    if (!(std::isfinite(mesh.multiblock_cart_core_bridge_spacing_floor) &&
          mesh.multiblock_cart_core_bridge_spacing_floor >= 0.0)) {
      throw ConfigError(
          "Mesh.multiblock_cart_core_bridge_spacing_floor must be finite "
          "and >= 0");
    }
    if (mesh.multiblock_cart_core_bridge_spacing_floor > 0.0 &&
        mesh.multiblock_cart_core_bridge_grading != "log") {
      throw ConfigError(
          "Mesh.multiblock_cart_core_bridge_spacing_floor > 0 requires "
          "Mesh.multiblock_cart_core_bridge_grading=\"log\"");
    }
  }
  if (has_key(kwargs, "multiblock_cart_core_bridge_ratio_max")) {
    mesh.multiblock_cart_core_bridge_ratio_max = numeric_as_double(
        kwargs["multiblock_cart_core_bridge_ratio_max"],
        "Mesh.multiblock_cart_core_bridge_ratio_max");
    if (!(std::isfinite(mesh.multiblock_cart_core_bridge_ratio_max) &&
          mesh.multiblock_cart_core_bridge_ratio_max > 1.0)) {
      throw ConfigError(
          "Mesh.multiblock_cart_core_bridge_ratio_max must be finite and "
          "> 1");
    }
    if (mesh.multiblock_cart_core_bridge_grading != "log") {
      throw ConfigError(
          "Mesh.multiblock_cart_core_bridge_ratio_max requires "
          "Mesh.multiblock_cart_core_bridge_grading=\"log\"");
    }
  }
  if (has_key(kwargs, "multiblock_theta_cap_widen_factor")) {
    mesh.multiblock_theta_cap_widen_factor = numeric_as_double(
        kwargs["multiblock_theta_cap_widen_factor"],
        "Mesh.multiblock_theta_cap_widen_factor");
    if (!(std::isfinite(mesh.multiblock_theta_cap_widen_factor) &&
          mesh.multiblock_theta_cap_widen_factor >= 1.0)) {
      throw ValueError(
          "Mesh.multiblock_theta_cap_widen_factor must be finite and >= 1");
    }
    if (mesh.multiblock_theta_cap_widen_factor != 1.0 &&
        mesh.topology_scheme !=
            TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL) {
      throw ConfigError(
          "multiblock_theta_cap_widen_factor requires "
          "multiblock_cart_core topology");
    }
    if (mesh.multiblock_theta_cap_widen_factor != 1.0 &&
        mesh.logical_mesh_2d == "polar_in_box") {
      throw ConfigError(
          "multiblock_theta_cap_widen_factor is not supported with "
          "polar_in_box (exterior theta ladder is equiangular)");
    }
  }
  if (has_key(kwargs, "multiblock_transition_scheme")) {
    mesh.multiblock_transition_scheme = parse_multiblock_transition_scheme(
        strict_string(kwargs["multiblock_transition_scheme"],
                      "Mesh.multiblock_transition_scheme"),
        "Mesh.multiblock_transition_scheme");
  }
  if (has_key(kwargs, "multiblock_cap_p")) {
    mesh.multiblock_cap_p =
        numeric_as_double(kwargs["multiblock_cap_p"],
                          "Mesh.multiblock_cap_p");
    if (!(std::isfinite(mesh.multiblock_cap_p) &&
          mesh.multiblock_cap_p > 2.0)) {
      throw ValueError(
          "Mesh.multiblock_cap_p must be finite and > 2");
    }
  }
  if (has_key(kwargs, "multiblock_bridge_elliptic_sweeps")) {
    mesh.multiblock_bridge_elliptic_sweeps =
        strict_int32(kwargs["multiblock_bridge_elliptic_sweeps"],
                     "Mesh.multiblock_bridge_elliptic_sweeps");
    ensure_int_ge(mesh.multiblock_bridge_elliptic_sweeps, 0,
                  "Mesh.multiblock_bridge_elliptic_sweeps");
  }
  if (has_key(kwargs, "multiblock_bridge_elliptic_omega")) {
    mesh.multiblock_bridge_elliptic_omega =
        numeric_as_double(kwargs["multiblock_bridge_elliptic_omega"],
                          "Mesh.multiblock_bridge_elliptic_omega");
    if (!(std::isfinite(mesh.multiblock_bridge_elliptic_omega) &&
          mesh.multiblock_bridge_elliptic_omega > 0.0 &&
          mesh.multiblock_bridge_elliptic_omega < 2.0)) {
      throw ValueError(
          "Mesh.multiblock_bridge_elliptic_omega must be finite and in (0, 2)");
    }
  }
  if (has_key(kwargs, "polar_tier_chi_lo")) {
    mesh.polar_tier_chi_lo =
        numeric_as_double(kwargs["polar_tier_chi_lo"],
                          "Mesh.polar_tier_chi_lo");
  }
  if (has_key(kwargs, "polar_tier_chi_hi")) {
    mesh.polar_tier_chi_hi =
        numeric_as_double(kwargs["polar_tier_chi_hi"],
                          "Mesh.polar_tier_chi_hi");
  }
  if (!(std::isfinite(mesh.polar_tier_chi_lo) &&
        std::isfinite(mesh.polar_tier_chi_hi) &&
        mesh.polar_tier_chi_lo > 0.0 &&
        mesh.polar_tier_chi_hi > mesh.polar_tier_chi_lo)) {
    throw ValueError(
        "Mesh.polar_tier_chi_lo/chi_hi must be finite with "
        "0 < chi_lo < chi_hi");
  }
  if (has_key(kwargs, "polar_tier_belt_thickness_frac")) {
    mesh.polar_tier_belt_thickness_frac =
        numeric_as_double(kwargs["polar_tier_belt_thickness_frac"],
                          "Mesh.polar_tier_belt_thickness_frac");
  }
  if (!(std::isfinite(mesh.polar_tier_belt_thickness_frac) &&
        mesh.polar_tier_belt_thickness_frac >= 0.0 &&
        mesh.polar_tier_belt_thickness_frac <= 0.9)) {
    throw ValueError(
        "Mesh.polar_tier_belt_thickness_frac must be finite and in [0, 0.9]");
  }
  if (has_key(kwargs, "polar_tier_belt_rows")) {
    mesh.polar_tier_belt_rows =
        strict_int32(kwargs["polar_tier_belt_rows"],
                     "Mesh.polar_tier_belt_rows");
  }
  if (mesh.polar_tier_belt_rows != 1 &&
      mesh.polar_tier_belt_rows != 2 &&
      mesh.polar_tier_belt_rows != 3) {
    throw ValueError(
        "Mesh.polar_tier_belt_rows must be one of {1, 2, 3}");
  }
  if (has_key(kwargs, "polar_tier_pole_cap_m")) {
    mesh.polar_tier_pole_cap_m =
        strict_int32(kwargs["polar_tier_pole_cap_m"],
                     "Mesh.polar_tier_pole_cap_m");
  }
  if (mesh.polar_tier_pole_cap_m != 0 &&
      (mesh.polar_tier_pole_cap_m < 4 ||
       mesh.polar_tier_pole_cap_m > 48)) {
    throw ValueError(
        "Mesh.polar_tier_pole_cap_m must be 0 or in [4, 48]");
  }
  if (has_key(kwargs, "polar_tier_pole_cap_alpha")) {
    mesh.polar_tier_pole_cap_alpha =
        numeric_as_double(kwargs["polar_tier_pole_cap_alpha"],
                          "Mesh.polar_tier_pole_cap_alpha");
  }
  if (!(std::isfinite(mesh.polar_tier_pole_cap_alpha) &&
        mesh.polar_tier_pole_cap_alpha >= 0.0 &&
        mesh.polar_tier_pole_cap_alpha <= 1.0)) {
    throw ValueError(
        "Mesh.polar_tier_pole_cap_alpha must be finite and in [0, 1]");
  }
  if (has_key(kwargs, "polar_tier_dendrite_enabled")) {
    mesh.polar_tier_dendrite_enabled =
        strict_bool(kwargs["polar_tier_dendrite_enabled"],
                    "Mesh.polar_tier_dendrite_enabled");
  }
  if (has_key(kwargs, "polar_tier_native_pentagon")) {
    mesh.polar_tier_native_pentagon =
        strict_bool(kwargs["polar_tier_native_pentagon"],
                    "Mesh.polar_tier_native_pentagon");
  }
  if (has_key(kwargs, "shell_polar_cap_dendrite")) {
    mesh.shell_polar_cap_dendrite =
        strict_bool(kwargs["shell_polar_cap_dendrite"],
                    "Mesh.shell_polar_cap_dendrite");
  }
  if (has_key(kwargs, "shell_cap_rows_2x")) {
    mesh.shell_cap_rows_2x =
        strict_int32(kwargs["shell_cap_rows_2x"],
                     "Mesh.shell_cap_rows_2x");
  }
  if (mesh.shell_cap_rows_2x < 12 || mesh.shell_cap_rows_2x > 288 ||
      mesh.shell_cap_rows_2x == 287) {
    throw ValueError(
        "Mesh.shell_cap_rows_2x must be in [12, 286] or equal 288");
  }
  if (has_key(kwargs, "polar_tier_dendrite_s_theta_rows_below")) {
    mesh.polar_tier_dendrite_s_theta_rows_below = strict_int32(
        kwargs["polar_tier_dendrite_s_theta_rows_below"],
        "Mesh.polar_tier_dendrite_s_theta_rows_below");
  }
  if (mesh.polar_tier_dendrite_s_theta_rows_below < 1 ||
      mesh.polar_tier_dendrite_s_theta_rows_below > 40) {
    throw ValueError(
        "Mesh.polar_tier_dendrite_s_theta_rows_below must be in [1, 40]");
  }
  if (has_key(kwargs, "polar_tier_fan_sectors")) {
    mesh.polar_tier_fan_sectors =
        strict_int32(kwargs["polar_tier_fan_sectors"],
                     "Mesh.polar_tier_fan_sectors");
  }
  if (mesh.polar_tier_fan_sectors != 6 &&
      mesh.polar_tier_fan_sectors != 12) {
    throw ValueError(
        "Mesh.polar_tier_fan_sectors must be 6 or 12");
  }
  if (has_key(kwargs, "polar_tier_min_tier_columns")) {
    mesh.polar_tier_min_tier_columns =
        strict_int32(kwargs["polar_tier_min_tier_columns"],
                     "Mesh.polar_tier_min_tier_columns");
    ensure_int_ge(mesh.polar_tier_min_tier_columns, 1,
                  "Mesh.polar_tier_min_tier_columns");
  }
  if (has_key(kwargs, "polar_tier_fan_first_ring_radius_cm")) {
    mesh.polar_tier_fan_first_ring_radius_cm =
        numeric_as_double(
            kwargs["polar_tier_fan_first_ring_radius_cm"],
            "Mesh.polar_tier_fan_first_ring_radius_cm");
  }
  if (!(std::isfinite(mesh.polar_tier_fan_first_ring_radius_cm) &&
        mesh.polar_tier_fan_first_ring_radius_cm >= 0.0)) {
    throw ValueError(
        "Mesh.polar_tier_fan_first_ring_radius_cm must be finite and >= 0");
  }
  if (has_key(kwargs, "polar_tier_hydro_enabled")) {
    mesh.polar_tier_hydro_enabled =
        strict_bool(kwargs["polar_tier_hydro_enabled"],
                    "Mesh.polar_tier_hydro_enabled");
  }
  validate_polar_tier_dendrite_config(mesh);
  if (has_key(kwargs, "multiblock_outer_svec_tangent_balance")) {
    mesh.multiblock_outer_svec_tangent_balance = strict_bool(
        kwargs["multiblock_outer_svec_tangent_balance"],
        "Mesh.multiblock_outer_svec_tangent_balance");
  }
  if (mesh.topology_scheme == TopologyScheme::MULTIBLOCK_POLAR_TIER ||
      mesh.topology_scheme ==
          TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER) {
    if (std::isnan(mesh.multiblock_cart_core_r_match)) {
      mesh.multiblock_cart_core_r_match =
          mesh.spherical_polar_s_max / 6.0;
    }
  } else if (mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK &&
             mesh.topology_scheme !=
                 TopologyScheme::PENTAGON_BELT_SHELL) {
    if (std::isnan(mesh.multiblock_cart_core_r_c)) {
      mesh.multiblock_cart_core_r_c = mesh.spherical_polar_s_max / 12.0;
    }
    if (std::isnan(mesh.multiblock_cart_core_r_match)) {
      mesh.multiblock_cart_core_r_match = 2.0 * mesh.multiblock_cart_core_r_c;
    }
    if (mesh.multiblock_cart_core_n_c < 0) {
      mesh.multiblock_cart_core_n_c = mesh.nz / 4;
    }
    if (mesh.multiblock_cart_core_bridge_layers < 0) {
      mesh.multiblock_cart_core_bridge_layers =
          std::max(4, static_cast<int>(
                          std::lround(static_cast<double>(
                                           mesh.multiblock_cart_core_n_c) /
                                       8.0)));
    }
  }
  if (has_key(kwargs, "floors")) {
    const py::handle floors_obj = kwargs["floors"];
    if (!py::isinstance<py::dict>(floors_obj)) {
      throw_value_type_error("Mesh.floors", "dict", floors_obj);
    }
    const py::dict floors = py::reinterpret_borrow<py::dict>(floors_obj);
    enforce_known_keys(floors, "Mesh.floors",
                       {"rho_floor_gcc", "Te_floor_eV", "Ti_floor_eV"});
    if (has_key(floors, "rho_floor_gcc")) {
      mesh.floors.rho_floor_gcc = numeric_as_double(floors["rho_floor_gcc"],
                                                    "Mesh.floors.rho_floor_gcc");
      ensure_non_negative(mesh.floors.rho_floor_gcc, "Mesh.floors.rho_floor_gcc");
      config.numerics.floors.rho = mesh.floors.rho_floor_gcc;
    }
    if (has_key(floors, "Te_floor_eV")) {
      mesh.floors.Te_floor_eV =
          numeric_as_double(floors["Te_floor_eV"], "Mesh.floors.Te_floor_eV");
      ensure_non_negative(mesh.floors.Te_floor_eV, "Mesh.floors.Te_floor_eV");
      config.numerics.floors.Te = mesh.floors.Te_floor_eV;
    }
    if (has_key(floors, "Ti_floor_eV")) {
      mesh.floors.Ti_floor_eV =
          numeric_as_double(floors["Ti_floor_eV"], "Mesh.floors.Ti_floor_eV");
      ensure_non_negative(mesh.floors.Ti_floor_eV, "Mesh.floors.Ti_floor_eV");
      config.numerics.floors.Ti = mesh.floors.Ti_floor_eV;
    }
  }

  if (has_key(kwargs, "auto_regions_axis")) {
    mesh.auto_regions_axis =
        strict_string(kwargs["auto_regions_axis"], "Mesh.auto_regions_axis");
    if (mesh.auto_regions_axis != "r" && mesh.auto_regions_axis != "z") {
      throw ValueError(
          "Mesh.auto_regions_axis must be one of {\"r\", \"z\"}, got " +
          mesh.auto_regions_axis);
    }
  }
  if (has_key(kwargs, "auto_regions")) {
    const py::handle regions_obj = kwargs["auto_regions"];
    if (!py::isinstance<py::list>(regions_obj) &&
        !py::isinstance<py::tuple>(regions_obj)) {
      throw_value_type_error("Mesh.auto_regions", "list|tuple", regions_obj);
    }
    const py::sequence region_seq =
        py::reinterpret_borrow<py::sequence>(regions_obj);
    if (region_seq.size() == 0) {
      throw ConfigError("Mesh.auto_regions must not be empty when provided");
    }
    double prev_r_end = -std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < region_seq.size(); ++k) {
      const py::handle item = region_seq[k];
      if (!py::isinstance<py::dict>(item)) {
        throw_value_type_error("Mesh.auto_regions[k]", "dict", item);
      }
      const py::dict region = py::reinterpret_borrow<py::dict>(item);
      enforce_known_keys(region, "Mesh.auto_regions[k]",
                         {"r_end", "nz", "rho_ref", "is_void",
                          "material_group"});
      Config::MeshConfig::AutoZoneRegion az;
      if (!has_key(region, "r_end") || !has_key(region, "nz") ||
          !has_key(region, "rho_ref")) {
        throw ConfigError(
            "Mesh.auto_regions entries require r_end, nz, rho_ref");
      }
      az.r_end = numeric_as_double(region["r_end"],
                                   "Mesh.auto_regions[k].r_end");
      az.nz = strict_int32(region["nz"], "Mesh.auto_regions[k].nz");
      az.rho_ref = numeric_as_double(region["rho_ref"],
                                     "Mesh.auto_regions[k].rho_ref");
      if (has_key(region, "is_void")) {
        az.is_void = strict_bool(region["is_void"],
                                 "Mesh.auto_regions[k].is_void");
      }
      if (has_key(region, "material_group")) {
        az.material_group = strict_string(
            region["material_group"], "Mesh.auto_regions[k].material_group");
      }
      if (!(az.r_end > prev_r_end)) {
        throw ConfigError(
            "Mesh.auto_regions r_end values must be strictly increasing");
      }
      if (az.nz <= 0) {
        throw ConfigError("Mesh.auto_regions nz must be positive");
      }
      if (!(az.rho_ref >= 0.0)) {
        throw ConfigError("Mesh.auto_regions rho_ref must be non-negative");
      }
      prev_r_end = az.r_end;
      mesh.auto_regions.push_back(az);
    }
  }
  if (has_key(kwargs, "zoning_intent")) {
    const py::handle intent_obj = kwargs["zoning_intent"];
    if (!py::isinstance<py::dict>(intent_obj)) {
      throw_value_type_error("Mesh.zoning_intent", "dict", intent_obj);
    }
    const py::dict intent = py::reinterpret_borrow<py::dict>(intent_obj);
    enforce_known_keys(
        intent, "Mesh.zoning_intent",
        {"n_cells", "measure", "pins", "profile", "anchors", "bands",
         "density_regions", "extra_events", "dr_min", "cell_measure_min",
         "cell_measure_max", "preferred_ratio", "ratio_hard_max",
         "min_cells_per_segment"});
    auto& zoning = mesh.zoning_intent;
    zoning.enabled = true;
    if (!has_key(intent, "n_cells")) {
      throw ConfigError("Mesh.zoning_intent requires key 'n_cells'");
    }
    zoning.n_cells =
        strict_int32(intent["n_cells"], "Mesh.zoning_intent.n_cells");
    if (zoning.n_cells < 1) {
      throw ConfigError("Mesh.zoning_intent.n_cells must be >= 1");
    }
    if (has_key(intent, "measure")) {
      zoning.measure =
          strict_string(intent["measure"], "Mesh.zoning_intent.measure");
      if (zoning.measure != "width" && zoning.measure != "areal_mass" &&
          zoning.measure != "cylindrical_line_mass" &&
          zoning.measure != "spherical_cell_mass") {
        throw ValueError(
            "Mesh.zoning_intent.measure must be one of {\"width\", "
            "\"areal_mass\", \"cylindrical_line_mass\", "
            "\"spherical_cell_mass\"}, got " +
            zoning.measure);
      }
    }
    if (has_key(intent, "pins")) {
      const py::handle pins_obj = intent["pins"];
      if (!py::isinstance<py::list>(pins_obj)) {
        throw_value_type_error("Mesh.zoning_intent.pins", "list[dict]",
                               pins_obj);
      }
      const py::list pins = py::reinterpret_borrow<py::list>(pins_obj);
      zoning.pins.reserve(pins.size());
      for (std::size_t k = 0; k < pins.size(); ++k) {
        const py::handle item = pins[k];
        if (!py::isinstance<py::dict>(item)) {
          throw_value_type_error("Mesh.zoning_intent.pins[k]", "dict", item);
        }
        const py::dict pin = py::reinterpret_borrow<py::dict>(item);
        enforce_known_keys(pin, "Mesh.zoning_intent.pins[k]",
                           {"r", "ratio_jump_allowed"});
        if (!has_key(pin, "r")) {
          throw ConfigError("Mesh.zoning_intent.pins entries require r");
        }
        Config::MeshConfig::ZoningIntentPinNL out;
        out.r = numeric_as_double(pin["r"], "Mesh.zoning_intent.pins[k].r");
        if (has_key(pin, "ratio_jump_allowed")) {
          out.ratio_jump_allowed = strict_bool(
              pin["ratio_jump_allowed"],
              "Mesh.zoning_intent.pins[k].ratio_jump_allowed");
        }
        zoning.pins.push_back(out);
      }
    }
    if (has_key(intent, "profile")) {
      const py::handle profile_obj = intent["profile"];
      if (!py::isinstance<py::list>(profile_obj)) {
        throw_value_type_error("Mesh.zoning_intent.profile", "list[dict]",
                               profile_obj);
      }
      const py::list profile = py::reinterpret_borrow<py::list>(profile_obj);
      zoning.profile.reserve(profile.size());
      for (std::size_t k = 0; k < profile.size(); ++k) {
        const py::handle item = profile[k];
        if (!py::isinstance<py::dict>(item)) {
          throw_value_type_error("Mesh.zoning_intent.profile[k]", "dict",
                                 item);
        }
        const py::dict point = py::reinterpret_borrow<py::dict>(item);
        enforce_known_keys(point, "Mesh.zoning_intent.profile[k]", {"r", "w"});
        if (!has_key(point, "r") || !has_key(point, "w")) {
          throw ConfigError("Mesh.zoning_intent.profile entries require r and w");
        }
        Config::MeshConfig::ZoningIntentProfilePointNL out;
        out.r = numeric_as_double(point["r"],
                                  "Mesh.zoning_intent.profile[k].r");
        out.w = numeric_as_double(point["w"],
                                  "Mesh.zoning_intent.profile[k].w");
        zoning.profile.push_back(out);
      }
    }
    if (has_key(intent, "anchors")) {
      const py::handle anchors_obj = intent["anchors"];
      if (!py::isinstance<py::list>(anchors_obj)) {
        throw_value_type_error("Mesh.zoning_intent.anchors", "list[dict]",
                               anchors_obj);
      }
      const py::list anchors = py::reinterpret_borrow<py::list>(anchors_obj);
      zoning.anchors.reserve(anchors.size());
      for (std::size_t k = 0; k < anchors.size(); ++k) {
        const py::handle item = anchors[k];
        if (!py::isinstance<py::dict>(item)) {
          throw_value_type_error("Mesh.zoning_intent.anchors[k]", "dict",
                                 item);
        }
        const py::dict anchor = py::reinterpret_borrow<py::dict>(item);
        enforce_known_keys(anchor, "Mesh.zoning_intent.anchors[k]",
                           {"r", "half_width", "log_amplitude"});
        if (!has_key(anchor, "r") || !has_key(anchor, "half_width") ||
            !has_key(anchor, "log_amplitude")) {
          throw ConfigError(
              "Mesh.zoning_intent.anchors entries require r, half_width, "
              "and log_amplitude");
        }
        Config::MeshConfig::ZoningIntentAnchorNL out;
        out.r = numeric_as_double(anchor["r"],
                                  "Mesh.zoning_intent.anchors[k].r");
        out.half_width = numeric_as_double(
            anchor["half_width"],
            "Mesh.zoning_intent.anchors[k].half_width");
        out.log_amplitude = numeric_as_double(
            anchor["log_amplitude"],
            "Mesh.zoning_intent.anchors[k].log_amplitude");
        zoning.anchors.push_back(out);
      }
    }
    if (has_key(intent, "bands")) {
      const py::handle bands_obj = intent["bands"];
      if (!py::isinstance<py::list>(bands_obj)) {
        throw_value_type_error("Mesh.zoning_intent.bands", "list[dict]",
                               bands_obj);
      }
      const py::list bands = py::reinterpret_borrow<py::list>(bands_obj);
      zoning.bands.reserve(bands.size());
      for (std::size_t k = 0; k < bands.size(); ++k) {
        const py::handle item = bands[k];
        if (!py::isinstance<py::dict>(item)) {
          throw_value_type_error("Mesh.zoning_intent.bands[k]", "dict", item);
        }
        const py::dict band = py::reinterpret_borrow<py::dict>(item);
        enforce_known_keys(
            band, "Mesh.zoning_intent.bands[k]",
            {"measure_frac_begin", "measure_frac_end", "cell_measure_min",
             "cell_measure_max"});
        if (!has_key(band, "measure_frac_begin") ||
            !has_key(band, "measure_frac_end")) {
          throw ConfigError(
              "Mesh.zoning_intent.bands entries require measure_frac_begin "
              "and measure_frac_end");
        }
        Config::MeshConfig::ZoningIntentBandNL out;
        out.measure_frac_begin = numeric_as_double(
            band["measure_frac_begin"],
            "Mesh.zoning_intent.bands[k].measure_frac_begin");
        out.measure_frac_end = numeric_as_double(
            band["measure_frac_end"],
            "Mesh.zoning_intent.bands[k].measure_frac_end");
        if (has_key(band, "cell_measure_min")) {
          out.cell_measure_min = numeric_as_double(
              band["cell_measure_min"],
              "Mesh.zoning_intent.bands[k].cell_measure_min");
        }
        if (has_key(band, "cell_measure_max")) {
          out.cell_measure_max = numeric_as_double(
              band["cell_measure_max"],
              "Mesh.zoning_intent.bands[k].cell_measure_max");
        }
        zoning.bands.push_back(out);
      }
    }
    if (has_key(intent, "density_regions")) {
      const py::handle density_obj = intent["density_regions"];
      if (!py::isinstance<py::list>(density_obj)) {
        throw_value_type_error("Mesh.zoning_intent.density_regions",
                               "list[dict]", density_obj);
      }
      const py::list regions = py::reinterpret_borrow<py::list>(density_obj);
      zoning.density_regions.reserve(regions.size());
      double prev_r_end = -std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < regions.size(); ++k) {
        const py::handle item = regions[k];
        if (!py::isinstance<py::dict>(item)) {
          throw_value_type_error("Mesh.zoning_intent.density_regions[k]",
                                 "dict", item);
        }
        const py::dict region = py::reinterpret_borrow<py::dict>(item);
        enforce_known_keys(region, "Mesh.zoning_intent.density_regions[k]",
                           {"r_end", "rho"});
        if (!has_key(region, "r_end") || !has_key(region, "rho")) {
          throw ConfigError(
              "Mesh.zoning_intent.density_regions entries require r_end "
              "and rho");
        }
        Config::MeshConfig::ZoningIntentDensityRegionNL out;
        out.r_end = numeric_as_double(
            region["r_end"],
            "Mesh.zoning_intent.density_regions[k].r_end");
        out.rho = numeric_as_double(
            region["rho"], "Mesh.zoning_intent.density_regions[k].rho");
        if (!(out.r_end > prev_r_end)) {
          throw ConfigError(
              "Mesh.zoning_intent.density_regions r_end values must be "
              "strictly increasing");
        }
        if (!(out.rho >= 0.0)) {
          throw ConfigError(
              "Mesh.zoning_intent.density_regions[k].rho must be >= 0");
        }
        prev_r_end = out.r_end;
        zoning.density_regions.push_back(out);
      }
    }
    if (has_key(intent, "extra_events")) {
      zoning.extra_events = strict_double_vector(
          intent["extra_events"], "Mesh.zoning_intent.extra_events");
    }
    if (has_key(intent, "dr_min")) {
      zoning.dr_min =
          numeric_as_double(intent["dr_min"], "Mesh.zoning_intent.dr_min");
    }
    if (has_key(intent, "cell_measure_min")) {
      zoning.cell_measure_min = numeric_as_double(
          intent["cell_measure_min"],
          "Mesh.zoning_intent.cell_measure_min");
    }
    if (has_key(intent, "cell_measure_max")) {
      zoning.cell_measure_max = numeric_as_double(
          intent["cell_measure_max"],
          "Mesh.zoning_intent.cell_measure_max");
    }
    if (has_key(intent, "preferred_ratio")) {
      zoning.preferred_ratio = numeric_as_double(
          intent["preferred_ratio"],
          "Mesh.zoning_intent.preferred_ratio");
    }
    if (has_key(intent, "ratio_hard_max")) {
      zoning.ratio_hard_max = numeric_as_double(
          intent["ratio_hard_max"],
          "Mesh.zoning_intent.ratio_hard_max");
    }
    if (has_key(intent, "min_cells_per_segment")) {
      zoning.min_cells_per_segment = strict_int32(
          intent["min_cells_per_segment"],
          "Mesh.zoning_intent.min_cells_per_segment");
    }
  }
  const auto parse_explicit_nodes = [&](const char* key,
                                        std::vector<double>& dst) {
    if (!has_key(kwargs, key)) {
      return;
    }
    const py::handle obj = kwargs[key];
    std::vector<double> nodes =
        strict_double_vector(obj, std::string("Mesh.") + key);
    if (nodes.size() < 2) {
      throw ValueError(std::string("Mesh.") + key +
                       " requires at least 2 nodes");
    }
    for (std::size_t i = 0; i + 1 < nodes.size(); ++i) {
      if (!(nodes[i] < nodes[i + 1])) {
        throw ValueError(std::string("Mesh.") + key +
                         " must be strictly increasing");
      }
    }
    for (const double v : nodes) {
      if (!std::isfinite(v)) {
        throw ValueError(std::string("Mesh.") + key +
                         " entries must be finite");
      }
    }
    dst = std::move(nodes);
  };
  parse_explicit_nodes("explicit_nodes", mesh.explicit_nodes);
  parse_explicit_nodes("explicit_nodes_z", mesh.explicit_nodes_z);
  parse_explicit_nodes("explicit_nodes_theta", mesh.explicit_nodes_theta);

  if (!mesh.explicit_nodes.empty()) {
    const int n = static_cast<int>(mesh.explicit_nodes.size()) - 1;
    const int expected_nr =
        mesh.logical_mesh_2d == "polar_in_box"
            ? n + mesh.morph_rings + mesh.collar_rings
            : n;
    if (mesh.nr >= 0 && mesh.nr != expected_nr) {
      throw ValueError("Mesh.nr conflicts with Mesh.explicit_nodes (" +
                       std::to_string(expected_nr) + " total zones)");
    }
    mesh.polar_prefix_nr =
        mesh.logical_mesh_2d == "polar_in_box" ? n : -1;
    mesh.nr = expected_nr;
  }
  if (!mesh.explicit_nodes_z.empty()) {
    const int n = static_cast<int>(mesh.explicit_nodes_z.size()) - 1;
    if (mesh.nz != 1 && mesh.nz != n) {
      throw ValueError("Mesh.nz conflicts with Mesh.explicit_nodes_z (" +
                       std::to_string(n) + " zones)");
    }
    mesh.nz = n;
  }

  const bool auto_axis_z = (mesh.auto_regions_axis == "z");
  if (!mesh.auto_regions.empty() &&
      ((auto_axis_z && !mesh.explicit_nodes_z.empty()) ||
       (!auto_axis_z && !mesh.explicit_nodes.empty()))) {
    throw ValueError(
        auto_axis_z
            ? "Mesh.auto_regions and Mesh.explicit_nodes_z are exclusive"
            : "Mesh.auto_regions and Mesh.explicit_nodes are exclusive");
  }
  if (has_key(kwargs, "grid_r") && !mesh.explicit_nodes.empty()) {
    throw ValueError("Mesh.grid_r and Mesh.explicit_nodes are exclusive");
  }
  if (has_key(kwargs, "grid_z") && !mesh.explicit_nodes_z.empty()) {
    throw ValueError("Mesh.grid_z and Mesh.explicit_nodes_z are exclusive");
  }
  // Mesh.grid_theta and Mesh.explicit_nodes_theta are rejected together by
  // the existing theta-zoning validation below.
  if (has_key(kwargs, "auto_zone")) {
    const py::handle az_obj = kwargs["auto_zone"];
    if (!py::isinstance<py::dict>(az_obj)) {
      throw_value_type_error("Mesh.auto_zone", "dict", az_obj);
    }
    const py::dict az = py::reinterpret_borrow<py::dict>(az_obj);
    enforce_known_keys(az, "Mesh.auto_zone",
                       {"mass_ratio_max", "n_bridge_min", "n_bridge_max",
                        "bridge_frac_max", "rho_void_cut", "dr_min",
                        "mass_ratio_hard_max", "max_iter", "bulk_mass_tol"});
    auto& cfg_az = mesh.auto_config;
    if (has_key(az, "mass_ratio_max")) {
      cfg_az.mass_ratio_max =
          numeric_as_double(az["mass_ratio_max"], "Mesh.auto_zone.mass_ratio_max");
      if (!(cfg_az.mass_ratio_max > 1.0)) {
        throw ConfigError("Mesh.auto_zone.mass_ratio_max must be > 1");
      }
    }
    if (has_key(az, "n_bridge_min")) {
      cfg_az.n_bridge_min = strict_int32(az["n_bridge_min"], "Mesh.auto_zone.n_bridge_min");
    }
    if (has_key(az, "n_bridge_max")) {
      cfg_az.n_bridge_max = strict_int32(az["n_bridge_max"], "Mesh.auto_zone.n_bridge_max");
    }
    if (has_key(az, "bridge_frac_max")) {
      cfg_az.bridge_frac_max =
          numeric_as_double(az["bridge_frac_max"], "Mesh.auto_zone.bridge_frac_max");
    }
    if (has_key(az, "rho_void_cut")) {
      cfg_az.rho_void_cut =
          numeric_as_double(az["rho_void_cut"], "Mesh.auto_zone.rho_void_cut");
    }
    if (has_key(az, "dr_min")) {
      cfg_az.dr_min = numeric_as_double(az["dr_min"], "Mesh.auto_zone.dr_min");
    }
    if (has_key(az, "mass_ratio_hard_max")) {
      cfg_az.mass_ratio_hard_max = numeric_as_double(
          az["mass_ratio_hard_max"], "Mesh.auto_zone.mass_ratio_hard_max");
    }
    if (has_key(az, "max_iter")) {
      cfg_az.max_iter = strict_int32(az["max_iter"], "Mesh.auto_zone.max_iter");
    }
    if (has_key(az, "bulk_mass_tol")) {
      cfg_az.bulk_mass_tol =
          numeric_as_double(az["bulk_mass_tol"], "Mesh.auto_zone.bulk_mass_tol");
    }
  }
  if (mesh.logical_mesh_2d == "cone_shell") {
    if (has_key(kwargs, "nr") || has_key(kwargs, "nz")) {
      throw ConfigError(
          "cone_shell derives Mesh.nr/Mesh.nz; omit nr and nz from the deck");
    }
    if (has_key(kwargs, "explicit_nodes") || has_key(kwargs, "grid_r") ||
        has_key(kwargs, "auto_regions") || has_key(kwargs, "box_center_z")) {
      throw ConfigError(
          "cone_shell is exclusive with Mesh.explicit_nodes, Mesh.grid_r, "
          "Mesh.auto_regions, and Mesh.box_center_z");
    }
  }
}

void Builder::set_materials(py::dict kwargs) {
  mark_block_called(Block::Materials);
  enforce_known_keys(kwargs, "Materials",
                     {"materials", "opacity_mix_rule",
                      "low_density_extrapolation", "mixture", "zbar",
                      "void_config"});

  auto& materials = config.materials;
  if (has_key(kwargs, "opacity_mix_rule")) {
    materials.opacity_mix_rule = strict_string(kwargs["opacity_mix_rule"],
                                               "Materials.opacity_mix_rule");
  }
  if (has_key(kwargs, "low_density_extrapolation")) {
    materials.low_density_extrapolation = strict_bool(
        kwargs["low_density_extrapolation"],
        "Materials.low_density_extrapolation");
  }

  if (has_key(kwargs, "mixture")) {
    const py::handle mixture_obj = kwargs["mixture"];
    if (!py::isinstance<py::dict>(mixture_obj)) {
      throw_value_type_error("Materials.mixture", "dict", mixture_obj);
    }
    const py::dict mixture = py::reinterpret_borrow<py::dict>(mixture_obj);
    enforce_known_keys(mixture, "Materials.mixture",
                       {"fractions", "eos_mix_rule", "opacity_mix_rule"});
    if (has_key(mixture, "opacity_mix_rule")) {
      materials.opacity_mix_rule = strict_string(
          mixture["opacity_mix_rule"], "Materials.mixture.opacity_mix_rule");
    }
    if (has_key(mixture, "fractions")) {
      warn_ignored_key("Materials.mixture.fractions");
    }
    if (has_key(mixture, "eos_mix_rule")) {
      warn_ignored_key("Materials.mixture.eos_mix_rule");
    }
  }

  if (has_key(kwargs, "zbar")) {
    const py::handle zbar_obj = kwargs["zbar"];
    if (!py::isinstance<py::dict>(zbar_obj)) {
      throw_value_type_error("Materials.zbar", "dict", zbar_obj);
    }
    const py::dict zbar = py::reinterpret_borrow<py::dict>(zbar_obj);
    enforce_known_keys(zbar, "Materials.zbar", {"model", "fixed_value", "table_file"});
    if (has_key(zbar, "model")) {
      materials.zbar.model = strict_string(zbar["model"], "Materials.zbar.model");
      if (!is_zbar_model(materials.zbar.model)) {
        throw ValueError(
            "Materials.zbar.model must be one of {\"fixed\", \"thomas_fermi\", \"tabular\"}, got " +
            materials.zbar.model);
      }
    }
    if (has_key(zbar, "fixed_value")) {
      materials.zbar.fixed_value =
          numeric_as_double(zbar["fixed_value"], "Materials.zbar.fixed_value");
    }
    if (has_key(zbar, "table_file")) {
      materials.zbar.table_file =
          strict_string(zbar["table_file"], "Materials.zbar.table_file");
    }
  }

  if (has_key(kwargs, "void_config")) {
    const py::handle void_config_obj = kwargs["void_config"];
    if (!py::isinstance<py::dict>(void_config_obj)) {
      throw_value_type_error("Materials.void_config", "dict", void_config_obj);
    }
    const py::dict void_config = py::reinterpret_borrow<py::dict>(void_config_obj);
    enforce_known_keys(void_config, "Materials.void_config", {"rho", "Te", "Ti"});
    if (has_key(void_config, "rho")) {
      materials.void_config.rho =
          numeric_as_double(void_config["rho"], "Materials.void_config.rho");
      ensure_non_negative(materials.void_config.rho, "Materials.void_config.rho");
    }
    if (has_key(void_config, "Te")) {
      materials.void_config.Te =
          numeric_as_double(void_config["Te"], "Materials.void_config.Te");
      ensure_non_negative(materials.void_config.Te, "Materials.void_config.Te");
    }
    if (has_key(void_config, "Ti")) {
      materials.void_config.Ti =
          numeric_as_double(void_config["Ti"], "Materials.void_config.Ti");
      ensure_non_negative(materials.void_config.Ti, "Materials.void_config.Ti");
    }
  }

  if (!has_key(kwargs, "materials")) {
    return;
  }

  const py::handle materials_obj = kwargs["materials"];
  if (!py::isinstance<py::sequence>(materials_obj) || py::isinstance<py::str>(materials_obj)) {
    throw_value_type_error("Materials.materials", "list[dict]", materials_obj);
  }

  const auto material_seq = py::reinterpret_borrow<py::sequence>(materials_obj);
  materials.materials.clear();
  materials.materials.reserve(material_seq.size());
  bool warned_default_eos_model = false;
  bool warned_default_kappa_a = false;
  bool warned_default_kappa_s = false;
  for (std::size_t i = 0; i < material_seq.size(); ++i) {
    const py::handle mat_obj = material_seq[i];
    if (!py::isinstance<py::dict>(mat_obj)) {
      throw_value_type_error("Materials.materials[" + std::to_string(i) + "]", "dict",
                             mat_obj);
    }
    const py::dict mat = py::reinterpret_borrow<py::dict>(mat_obj);
    enforce_known_keys(mat, "Materials.materials",
                       {"name", "A", "Z", "eos", "opacity", "is_void"});

    Config::MaterialsConfig::MatDef def;
    bool kappa_a_explicit = false;
    bool kappa_s_explicit = false;
    if (has_key(mat, "name")) {
      def.name = strict_string(mat["name"],
                               "Materials.materials[" + std::to_string(i) + "].name");
    }
    if (has_key(mat, "A")) {
      def.A = numeric_as_double(mat["A"],
                                "Materials.materials[" + std::to_string(i) + "].A");
    }
    if (has_key(mat, "Z")) {
      def.Z = numeric_as_double(mat["Z"],
                                "Materials.materials[" + std::to_string(i) + "].Z");
    }
    if (has_key(mat, "is_void")) {
      def.is_void = strict_bool(mat["is_void"],
                                "Materials.materials[" + std::to_string(i) + "].is_void");
    }

    bool eos_model_explicit = false;
    if (has_key(mat, "eos")) {
      const py::handle eos_obj = mat["eos"];
      if (!py::isinstance<py::dict>(eos_obj)) {
        throw_value_type_error("Materials.materials[" + std::to_string(i) + "].eos", "dict",
                               eos_obj);
      }
      const py::dict eos = py::reinterpret_borrow<py::dict>(eos_obj);
      enforce_known_keys(eos, "Materials.materials.eos",
                         {"model", "file", "sesame_material_id", "ideal_gas",
                          "cv_e_override", "eos_T_ref_eV", "hydro_backend",
                          "mg_T_ref_eV", "mg_dT_rel", "f_erg_g", "beta",
                          "mu_rho", "gamma_p", "step_D_erg_g_eV",
                          "step_Tc_eV", "step_w_eV"});
      if (has_key(eos, "model")) {
        def.eos_model = strict_string(
            eos["model"],
            "Materials.materials[" + std::to_string(i) + "].eos.model");
        eos_model_explicit = true;
        if (!is_eos_model(def.eos_model)) {
          throw ValueError(
              "Materials.materials[" + std::to_string(i) +
              "].eos.model must be one of {\"sesame\", \"ionmix\", \"tmat\", \"ideal_gas\", \"power_law_te\"}, got " +
              def.eos_model);
        }
      }
      if (has_key(eos, "file")) {
        const std::string raw_eos_file = strict_string(
            eos["file"], "Materials.materials[" + std::to_string(i) + "].eos.file");
        def.eos_file = resolve_namelist_relative_path(config, raw_eos_file);
      }
      if (has_key(eos, "sesame_material_id")) {
        def.sesame_material_id = strict_int32(
            eos["sesame_material_id"],
            "Materials.materials[" + std::to_string(i) + "].eos.sesame_material_id");
      }
      if (has_key(eos, "cv_e_override")) {
        def.cv_e_override = numeric_as_double(
            eos["cv_e_override"],
            "Materials.materials[" + std::to_string(i) + "].eos.cv_e_override");
      }
      if (has_key(eos, "eos_T_ref_eV")) {
        def.eos_T_ref_eV = numeric_as_double(
            eos["eos_T_ref_eV"],
            "Materials.materials[" + std::to_string(i) + "].eos.eos_T_ref_eV");
      }
      if (has_key(eos, "hydro_backend")) {
        def.hydro_eos_backend = strict_string(
            eos["hydro_backend"],
            "Materials.materials[" + std::to_string(i) + "].eos.hydro_backend");
        if (!is_hydro_eos_backend(def.hydro_eos_backend)) {
          throw ValueError(
              "Materials.materials[" + std::to_string(i) +
              "].eos.hydro_backend must be one of {\"legacy\", \"helmholtz_spline\", \"helmholtz_jet\", \"exact_ideal_gas\", \"rho_e_table\", \"mie_gruneisen\"}, got " +
              def.hydro_eos_backend);
        }
      }
      if (has_key(eos, "mg_T_ref_eV")) {
        def.mg_T_ref_eV = numeric_as_double(
            eos["mg_T_ref_eV"],
            "Materials.materials[" + std::to_string(i) + "].eos.mg_T_ref_eV");
      }
      if (has_key(eos, "mg_dT_rel")) {
        def.mg_dT_rel = numeric_as_double(
            eos["mg_dT_rel"],
            "Materials.materials[" + std::to_string(i) + "].eos.mg_dT_rel");
      }
      if (has_key(eos, "ideal_gas")) {
        const py::handle ig_obj = eos["ideal_gas"];
        if (!py::isinstance<py::dict>(ig_obj)) {
          throw_value_type_error(
              "Materials.materials[" + std::to_string(i) + "].eos.ideal_gas", "dict",
              ig_obj);
        }
        const py::dict ig = py::reinterpret_borrow<py::dict>(ig_obj);
        enforce_known_keys(ig, "Materials.materials.eos.ideal_gas", {"gamma"});
        if (has_key(ig, "gamma")) {
          def.ideal_gas_gamma = numeric_as_double(
              ig["gamma"],
              "Materials.materials[" + std::to_string(i) + "].eos.ideal_gas.gamma");
        }
      }
      if (def.eos_model == "power_law_te") {
        if (has_key(eos, "ideal_gas")) {
          throw ConfigError(
              "Materials.materials[" + std::to_string(i) +
              "].eos.ideal_gas is not valid for eos.model=\"power_law_te\"");
        }
        if (!has_key(eos, "f_erg_g")) {
          throw ConfigError("Materials.materials[" + std::to_string(i) +
                            "].eos.power_law_te requires key f_erg_g");
        }
        def.eos_power_law_f_erg_g = numeric_as_double(
            eos["f_erg_g"],
            "Materials.materials[" + std::to_string(i) + "].eos.f_erg_g");
        if (!(def.eos_power_law_f_erg_g > 0.0)) {
          throw ConfigError("Materials.materials[" + std::to_string(i) +
                            "].eos.f_erg_g must be > 0");
        }
        if (!has_key(eos, "beta")) {
          throw ConfigError("Materials.materials[" + std::to_string(i) +
                            "].eos.power_law_te requires key beta");
        }
        def.eos_power_law_beta = numeric_as_double(
            eos["beta"],
            "Materials.materials[" + std::to_string(i) + "].eos.beta");
        if (!(def.eos_power_law_beta > 0.0)) {
          throw ConfigError("Materials.materials[" + std::to_string(i) +
                            "].eos.beta must be > 0");
        }
        if (has_key(eos, "mu_rho")) {
          def.eos_power_law_mu_rho = numeric_as_double(
              eos["mu_rho"],
              "Materials.materials[" + std::to_string(i) + "].eos.mu_rho");
        }
        if (has_key(eos, "step_D_erg_g_eV")) {
          def.eos_power_law_step_D_erg_g_eV = numeric_as_double(
              eos["step_D_erg_g_eV"],
              "Materials.materials[" + std::to_string(i) +
                  "].eos.step_D_erg_g_eV");
          if (!(def.eos_power_law_step_D_erg_g_eV >= 0.0)) {
            throw ConfigError("Materials.materials[" + std::to_string(i) +
                              "].eos.step_D_erg_g_eV must be >= 0");
          }
        }
        if (has_key(eos, "step_Tc_eV")) {
          def.eos_power_law_step_Tc_eV = numeric_as_double(
              eos["step_Tc_eV"],
              "Materials.materials[" + std::to_string(i) + "].eos.step_Tc_eV");
        }
        if (has_key(eos, "step_w_eV")) {
          def.eos_power_law_step_w_eV = numeric_as_double(
              eos["step_w_eV"],
              "Materials.materials[" + std::to_string(i) + "].eos.step_w_eV");
        }
        if (def.eos_power_law_step_D_erg_g_eV > 0.0 &&
            (!(def.eos_power_law_step_Tc_eV > 0.0) ||
             !(def.eos_power_law_step_w_eV > 0.0))) {
          throw ConfigError(
              "Materials.materials[" + std::to_string(i) +
              "].eos: step_D_erg_g_eV > 0 requires step_Tc_eV > 0 and"
              " step_w_eV > 0");
        }
        if (has_key(eos, "gamma_p")) {
          def.eos_power_law_gamma_p = numeric_as_double(
              eos["gamma_p"],
              "Materials.materials[" + std::to_string(i) + "].eos.gamma_p");
        }
        if (!(def.eos_power_law_gamma_p > 1.0)) {
          throw ConfigError("Materials.materials[" + std::to_string(i) +
                            "].eos.gamma_p must be > 1");
        }
      }
    }
    if (!eos_model_explicit && !def.is_void) {
      if (def.eos_model == "ideal_gas" && !warned_default_eos_model) {
        std::cerr << "[WARN] eos.model defaults to ideal_gas\n";
        warned_default_eos_model = true;
      }
    }

    if (has_key(mat, "opacity")) {
      const py::handle opacity_obj = mat["opacity"];
      if (!py::isinstance<py::dict>(opacity_obj)) {
        throw_value_type_error("Materials.materials[" + std::to_string(i) + "].opacity",
                               "dict", opacity_obj);
      }
      const py::dict opacity = py::reinterpret_borrow<py::dict>(opacity_obj);
      enforce_known_keys(opacity, "Materials.materials.opacity",
                         {"model", "file", "tmat_skip_lte_repair",
                          "kappa_a", "kappa_planck", "kappa_s", "units",
                          "lambda_method", "lambda_fd_delta_rel",
                          "lambda_fd_abs_min", "f_min", "kappa0_cm2_g",
                          "alpha_T", "lambda_rho", "T_ref_eV",
                          "rho_ref_g_cc"});
      if (has_key(opacity, "model")) {
        def.opacity_model = strict_string(
            opacity["model"],
            "Materials.materials[" + std::to_string(i) + "].opacity.model");
        if (!is_opacity_model(def.opacity_model)) {
          throw ValueError(
              "Materials.materials[" + std::to_string(i) +
              "].opacity.model must be one of {\"ionmix\", \"sesame\", \"constant\", \"freq_dep_marshak\", \"table_nlte\", \"tmat\", \"power_law\", \"none\"}, got " +
              def.opacity_model);
        }
      }
      if (has_key(opacity, "file")) {
        const std::string raw_opacity_file = strict_string(
            opacity["file"], "Materials.materials[" + std::to_string(i) + "].opacity.file");
        def.opacity_file = resolve_namelist_relative_path(config, raw_opacity_file);
      }
      if (has_key(opacity, "tmat_skip_lte_repair")) {
        def.tmat_skip_lte_repair = strict_bool(
            opacity["tmat_skip_lte_repair"],
            "Materials.materials[" + std::to_string(i) +
                "].opacity.tmat_skip_lte_repair");
      }
      if (has_key(opacity, "kappa_a")) {
        kappa_a_explicit = true;
        def.kappa_a_constant =
            numeric_as_double(opacity["kappa_a"],
                              "Materials.materials[" + std::to_string(i) + "].opacity.kappa_a");
      }
      if (has_key(opacity, "kappa_planck")) {
        const std::string path = "Materials.materials[" + std::to_string(i) +
                                 "].opacity.kappa_planck";
        try {
          def.kappa_planck_override =
              numeric_as_double(opacity["kappa_planck"], path);
        } catch (const ValueError& error) {
          throw ConfigError(error.what());
        }
        if (!(def.kappa_planck_override >= 0.0)) {
          throw ConfigError(format_range_error(
              path, ">= 0 (finite)", std::to_string(def.kappa_planck_override)));
        }
      }
      if (has_key(opacity, "kappa_s")) {
        kappa_s_explicit = true;
        def.kappa_s_constant =
            numeric_as_double(opacity["kappa_s"],
                              "Materials.materials[" + std::to_string(i) + "].opacity.kappa_s");
      }
      if (has_key(opacity, "units")) {
        def.opacity_units =
            strict_string(opacity["units"],
                          "Materials.materials[" + std::to_string(i) + "].opacity.units");
      }
      if (has_key(opacity, "lambda_method")) {
        def.lambda_method = strict_string(
            opacity["lambda_method"],
            "Materials.materials[" + std::to_string(i) + "].opacity.lambda_method");
        if (!is_lambda_method(def.lambda_method)) {
          throw ValueError(
              "Materials.materials[" + std::to_string(i) +
              "].opacity.lambda_method must be one of {\"finite_difference\", \"freeze_opacity\"}, got " +
              def.lambda_method);
        }
      }
      if (has_key(opacity, "lambda_fd_delta_rel")) {
        def.lambda_fd_delta_rel = numeric_as_double(
            opacity["lambda_fd_delta_rel"],
            "Materials.materials[" + std::to_string(i) +
                "].opacity.lambda_fd_delta_rel");
      }
      if (has_key(opacity, "lambda_fd_abs_min")) {
        def.lambda_fd_abs_min = numeric_as_double(
            opacity["lambda_fd_abs_min"],
            "Materials.materials[" + std::to_string(i) +
                "].opacity.lambda_fd_abs_min");
      }
      if (has_key(opacity, "f_min")) {
        def.nlte_f_min = numeric_as_double(
            opacity["f_min"],
            "Materials.materials[" + std::to_string(i) + "].opacity.f_min");
      }
      if (def.opacity_model == "power_law") {
        if (!has_key(opacity, "kappa0_cm2_g")) {
          throw ConfigError("Materials.materials[" + std::to_string(i) +
                            "].opacity.power_law requires key kappa0_cm2_g");
        }
        def.opacity_power_law_kappa0_cm2_g = numeric_as_double(
            opacity["kappa0_cm2_g"],
            "Materials.materials[" + std::to_string(i) + "].opacity.kappa0_cm2_g");
        if (!(def.opacity_power_law_kappa0_cm2_g > 0.0)) {
          throw ConfigError("Materials.materials[" + std::to_string(i) +
                            "].opacity.kappa0_cm2_g must be > 0");
        }
        if (has_key(opacity, "alpha_T")) {
          def.opacity_power_law_alpha_T = numeric_as_double(
              opacity["alpha_T"],
              "Materials.materials[" + std::to_string(i) + "].opacity.alpha_T");
        }
        if (has_key(opacity, "lambda_rho")) {
          def.opacity_power_law_lambda_rho = numeric_as_double(
              opacity["lambda_rho"],
              "Materials.materials[" + std::to_string(i) + "].opacity.lambda_rho");
        }
        if (has_key(opacity, "T_ref_eV")) {
          def.opacity_power_law_T_ref_eV = numeric_as_double(
              opacity["T_ref_eV"],
              "Materials.materials[" + std::to_string(i) + "].opacity.T_ref_eV");
        }
        if (!(def.opacity_power_law_T_ref_eV > 0.0)) {
          throw ConfigError("Materials.materials[" + std::to_string(i) +
                            "].opacity.T_ref_eV must be > 0");
        }
        if (has_key(opacity, "rho_ref_g_cc")) {
          def.opacity_power_law_rho_ref_g_cc = numeric_as_double(
              opacity["rho_ref_g_cc"],
              "Materials.materials[" + std::to_string(i) + "].opacity.rho_ref_g_cc");
        }
        if (!(def.opacity_power_law_rho_ref_g_cc > 0.0)) {
          throw ConfigError("Materials.materials[" + std::to_string(i) +
                            "].opacity.rho_ref_g_cc must be > 0");
        }
      }
    }
    if (def.is_void) {
      def.eos_model = "ideal_gas";
      def.opacity_model = "constant";
      def.kappa_a_constant = 0.0;
      def.kappa_s_constant = 0.0;
      if (def.A == 0.0) {
        def.A = 1.0;
      }
      def.Z = 0.0;
    }
    if (def.opacity_model != "table_nlte" && def.opacity_model != "tmat") {
      if (def.lambda_method != "finite_difference" || def.lambda_fd_delta_rel != 1.0e-4 ||
          def.lambda_fd_abs_min != 1.0e-6 || def.nlte_f_min != 1.0e-4) {
        std::cerr << "[WARN] NLTE knobs (lambda_method/fd_delta_rel/fd_abs_min/f_min) "
                     "are set but opacity.model is not table_nlte/tmat; they will be ignored\n";
      }
    }

    if (def.opacity_units != "cm2_per_g") {
      const std::string mat_id =
          def.name.empty() ? ("#" + std::to_string(i)) : ("\"" + def.name + "\"");
      throw ConfigError("Materials.materials[" + mat_id +
                        "].opacity.units must be \"cm2_per_g\" in v1.0");
    }

    if (def.opacity_model == "none") {
      tenryu::core::log_warning("Materials.opacity.model=\"none\" is deprecated; converted to "
                                "\"constant\" with kappa_a=kappa_s=0");
      def.opacity_model = "constant";
      def.kappa_a_constant = 0.0;
      def.kappa_s_constant = 0.0;
    }
    if (def.opacity_model == "constant" && !def.is_void) {
      if (!kappa_a_explicit && def.kappa_a_constant == 0.0 && !warned_default_kappa_a) {
        std::cerr << "[WARN] opacity.kappa_a defaults to 0 (transparent medium)\n";
        warned_default_kappa_a = true;
      }
      if (!kappa_s_explicit && def.kappa_s_constant == 0.0 && !warned_default_kappa_s) {
        std::cerr << "[WARN] opacity.kappa_s defaults to 0 (no scattering)\n";
        warned_default_kappa_s = true;
      }
    }

    materials.materials.push_back(std::move(def));
  }
}

void Builder::set_geometry(py::dict kwargs) {
  mark_block_called(Block::Geometry);
  enforce_known_keys(kwargs, "Geometry",
                     {"rho", "Te", "Ti", "velocity", "volfrac", "radiation_field",
                      "radiation_field_Tr_eV", "enforce_sum_to_one"});

  auto& geometry = config.geometry;
  if (has_key(kwargs, "rho")) {
    const auto callable = extract_callable_or_throw(kwargs["rho"], "Geometry.rho");
    geometry.rho = to_config_callable(callable);
    register_callable("Geometry.rho", callable, kwargs["rho"]);
  }
  if (has_key(kwargs, "Te")) {
    const auto callable = extract_callable_or_throw(kwargs["Te"], "Geometry.Te");
    geometry.Te = to_config_callable(callable);
    register_callable("Geometry.Te", callable, kwargs["Te"]);
  }
  if (has_key(kwargs, "Ti")) {
    const auto callable = extract_callable_or_throw(kwargs["Ti"], "Geometry.Ti");
    geometry.Ti = to_config_callable(callable);
    register_callable("Geometry.Ti", callable, kwargs["Ti"]);
  }
  if (has_key(kwargs, "velocity") && !kwargs["velocity"].is_none()) {
    const auto callable =
        extract_callable_or_throw(kwargs["velocity"], "Geometry.velocity");
    geometry.velocity = to_config_callable(callable);
    register_callable("Geometry.velocity", callable, kwargs["velocity"]);
  }
  if (has_key(kwargs, "radiation_field")) {
    const std::string value =
        strict_string(kwargs["radiation_field"], "Geometry.radiation_field");
    if (value != "equilibrium" && value != "zero" && value != "planck") {
      throw ConfigError(
          "Geometry.radiation_field must be 'equilibrium', 'zero', or 'planck', got '" +
          value + "'");
    }
    geometry.radiation_field = value;
  }
  if (has_key(kwargs, "radiation_field_Tr_eV")) {
    geometry.radiation_field_Tr_eV = numeric_as_double(
        kwargs["radiation_field_Tr_eV"], "Geometry.radiation_field_Tr_eV");
  }
  if (geometry.radiation_field == "planck" &&
      !(geometry.radiation_field_Tr_eV > 0.0)) {
    throw ConfigError(
        "Geometry.radiation_field == 'planck' requires "
        "Geometry.radiation_field_Tr_eV > 0");
  }
  if (has_key(kwargs, "enforce_sum_to_one")) {
    geometry.enforce_sum_to_one =
        strict_bool(kwargs["enforce_sum_to_one"], "Geometry.enforce_sum_to_one");
  }
  if (has_key(kwargs, "volfrac")) {
    const py::handle volfrac_obj = kwargs["volfrac"];
    if (!py::isinstance<py::dict>(volfrac_obj)) {
      throw_value_type_error("Geometry.volfrac", "dict", volfrac_obj);
    }
    const py::dict volfrac = py::reinterpret_borrow<py::dict>(volfrac_obj);
    geometry.volfrac.clear();
    for (const auto item : volfrac) {
      const std::string key = py::str(item.first).cast<std::string>();
      const auto callable =
          extract_callable_or_throw(item.second, "Geometry.volfrac." + key);
      geometry.volfrac[key] = to_config_callable(callable);
      register_callable("Geometry.volfrac." + key, callable, item.second);
    }
  }
}

void Builder::set_radiation(py::dict kwargs) {
  mark_block_called(Block::Radiation);
  enforce_known_keys(kwargs, "Radiation",
                     {"enabled", "origin_parity_only", "group_repack_hard_xray",
                      "diagnose_hard_xray_opacity",
                      "mode", "groups", "group_bounds_eV", "imc", "ddmc", "diffusion",
                      "multigroup_diffusion", "sn_transport", "holo", "boundary",
                      "max_pool_size", "momentum_deposition",
                      "volume_source_rate", "volume_source_x_max",
                      "compute_T_range_eV"});

  auto& radiation = config.radiation;
  const auto apply_boundary_type_to_faces = [&](const std::string& type) {
    radiation.boundary.inner_r = type;
    radiation.boundary.outer_r = type;
    radiation.boundary.bottom_z = type;
    radiation.boundary.top_z = type;
  };
  bool use_log_uniform_bounds = false;
  const bool mode_explicit = has_key(kwargs, "mode");
  const bool imc_explicit = has_key(kwargs, "imc");
  const bool ddmc_explicit = has_key(kwargs, "ddmc");
  if (has_key(kwargs, "enabled")) {
    radiation.enabled = strict_bool(kwargs["enabled"], "Radiation.enabled");
  }
  if (mode_explicit) {
    const std::string mode = strict_string(kwargs["mode"], "Radiation.mode");
    if (mode == "imc_ddmc") {
      radiation.mode = RadiationMode::ImcDdmc;
    } else if (mode == "multigroup_diffusion") {
      radiation.mode = RadiationMode::MultigroupDiffusion;
    } else if (mode == "sn_transport") {
      radiation.mode = RadiationMode::SnTransport;
    } else {
      throw ConfigError(
          "Radiation.mode must be \"imc_ddmc\", \"multigroup_diffusion\", or \"sn_transport\"");
    }
  }
  if (!mode_explicit && radiation.mode == RadiationMode::MultigroupDiffusion) {
    if (!imc_explicit) {
      radiation.imc.enabled = false;
    }
    if (!ddmc_explicit) {
      radiation.ddmc.enabled = false;
    }
  }
  if (has_key(kwargs, "origin_parity_only")) {
    radiation.origin_parity_only =
        strict_bool(kwargs["origin_parity_only"], "Radiation.origin_parity_only");
  }
  if (has_key(kwargs, "group_repack_hard_xray")) {
    radiation.group_repack_hard_xray = strict_bool(
        kwargs["group_repack_hard_xray"], "Radiation.group_repack_hard_xray");
  }
  if (has_key(kwargs, "diagnose_hard_xray_opacity")) {
    radiation.diagnose_hard_xray_opacity = strict_bool(
        kwargs["diagnose_hard_xray_opacity"], "Radiation.diagnose_hard_xray_opacity");
  }
  if (has_key(kwargs, "max_pool_size")) {
    warn_ignored_key("Radiation.max_pool_size");
  }
  if (has_key(kwargs, "momentum_deposition")) {
    warn_ignored_key("Radiation.momentum_deposition");
  }

  if (has_key(kwargs, "compute_T_range_eV")) {
    auto t_range = strict_double_vector(kwargs["compute_T_range_eV"],
                                        "Radiation.compute_T_range_eV");
    if (t_range.size() != 2) {
      throw ValueError("Radiation.compute_T_range_eV must be list[2]");
    }
    radiation.compute_T_range_eV = t_range;
    radiation.planck_fraction.compute_T_range_eV = std::move(t_range);
  }

  if (has_key(kwargs, "group_bounds_eV")) {
    const py::handle bounds_obj = kwargs["group_bounds_eV"];
    if (py::isinstance<py::str>(bounds_obj)) {
      const std::string mode =
          strict_string(bounds_obj, "Radiation.group_bounds_eV");
      if (mode == "log_uniform") {
        use_log_uniform_bounds = true;
        radiation.group_bounds_eV.clear();
      } else {
        throw ValueError(
            "Radiation.group_bounds_eV must be list[float] or \"log_uniform\"");
      }
    } else {
      radiation.group_bounds_eV =
          strict_double_vector(bounds_obj, "Radiation.group_bounds_eV");
      use_log_uniform_bounds = false;
    }
  }
  if (has_key(kwargs, "groups")) {
    const py::handle groups_obj = kwargs["groups"];
    if (py::isinstance<py::int_>(groups_obj) && !py::isinstance<py::bool_>(groups_obj)) {
      radiation.groups = strict_int32(groups_obj, "Radiation.groups");
    } else if (py::isinstance<py::dict>(groups_obj)) {
      const py::dict groups = py::reinterpret_borrow<py::dict>(groups_obj);
      enforce_known_keys(groups, "Radiation.groups",
                         {"bounds_eV", "representative", "planck_fraction"});
      if (has_key(groups, "bounds_eV")) {
        const py::handle bounds_obj = groups["bounds_eV"];
        if (py::isinstance<py::str>(bounds_obj)) {
          const std::string mode =
              strict_string(bounds_obj, "Radiation.groups.bounds_eV");
          if (mode == "log_uniform") {
            use_log_uniform_bounds = true;
            radiation.group_bounds_eV.clear();
          } else {
            throw ValueError(
                "Radiation.groups.bounds_eV must be list[float] or \"log_uniform\"");
          }
        } else {
          radiation.group_bounds_eV = strict_double_vector(
              bounds_obj, "Radiation.groups.bounds_eV");
          use_log_uniform_bounds = false;
        }
      }
      if (has_key(groups, "planck_fraction")) {
        const py::handle pf_obj = groups["planck_fraction"];
        if (!py::isinstance<py::dict>(pf_obj)) {
          throw_value_type_error("Radiation.groups.planck_fraction", "dict", pf_obj);
        }
        const py::dict pf = py::reinterpret_borrow<py::dict>(pf_obj);
        enforce_known_keys(pf, "Radiation.groups.planck_fraction",
                           {"method", "compute_N_T", "compute_T_range_eV",
                            "T_grid_eV", "b_g"});
        if (has_key(pf, "method")) {
          radiation.planck_fraction.method =
              strict_string(pf["method"], "Radiation.groups.planck_fraction.method");
        }
        if (has_key(pf, "compute_N_T")) {
          radiation.planck_fraction.compute_N_T = strict_int32(
              pf["compute_N_T"], "Radiation.groups.planck_fraction.compute_N_T");
        }
        if (has_key(pf, "compute_T_range_eV")) {
          auto t_range = strict_double_vector(
              pf["compute_T_range_eV"], "Radiation.groups.planck_fraction.compute_T_range_eV");
          if (t_range.size() != 2) {
            throw ValueError(
                "Radiation.groups.planck_fraction.compute_T_range_eV must be list[2]");
          }
          radiation.compute_T_range_eV = t_range;
          radiation.planck_fraction.compute_T_range_eV = std::move(t_range);
        }
        if (has_key(pf, "T_grid_eV")) {
          radiation.planck_fraction.T_grid_eV = strict_double_vector(
              pf["T_grid_eV"], "Radiation.groups.planck_fraction.T_grid_eV");
        }
        if (has_key(pf, "b_g")) {
          const py::handle bg_obj = pf["b_g"];
          if (!py::isinstance<py::sequence>(bg_obj) || py::isinstance<py::str>(bg_obj)) {
            throw_value_type_error("Radiation.groups.planck_fraction.b_g", "list[list[float]]",
                                   bg_obj);
          }
          const py::sequence bg_rows = py::reinterpret_borrow<py::sequence>(bg_obj);
          radiation.planck_fraction.b_g.clear();
          radiation.planck_fraction.b_g.reserve(bg_rows.size());
          for (std::size_t row = 0; row < bg_rows.size(); ++row) {
            radiation.planck_fraction.b_g.push_back(strict_double_vector(
                bg_rows[row], "Radiation.groups.planck_fraction.b_g[" + std::to_string(row) + "]"));
          }
        }
      }
    } else {
      throw_value_type_error("Radiation.groups", "int|dict", groups_obj);
    }
  }

  if (use_log_uniform_bounds) {
    if (valid_temperature_range(radiation.compute_T_range_eV)) {
      const double Tmin = radiation.compute_T_range_eV[0];
      const double Tmax = radiation.compute_T_range_eV[1];
      if (radiation.groups <= 0) {
        throw ConfigError(
            "Radiation.groups must be > 0 when using log_uniform group bounds");
      }
      if (radiation.group_repack_hard_xray) {
        radiation.group_bounds_eV =
            tenryu::core::repack_radiation_group_bounds_for_hard_xray(
                radiation.groups, radiation.compute_T_range_eV);
        if (radiation.group_bounds_eV.empty()) {
          throw ConfigError("Radiation.group_repack_hard_xray failed to produce group bounds");
        }
        const int hard_xray_groups =
            tenryu::core::count_radiation_groups_inside_energy_band(
                radiation.group_bounds_eV, 2000.0, 5000.0);
        if (hard_xray_groups < 20) {
          throw ConfigError(
              "Radiation.group_repack_hard_xray produced fewer than 20 groups "
              "inside 2-5 keV");
        }
      } else {
        radiation.group_bounds_eV =
            make_log_uniform_bounds(radiation.groups, Tmin, Tmax);
      }
    }
  }

  if (!radiation.group_bounds_eV.empty()) {
    const std::size_t bounds_size = radiation.group_bounds_eV.size();
    constexpr std::size_t kMaxGroupBoundsSize =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1U;
    if (bounds_size > kMaxGroupBoundsSize) {
      throw ConfigError("Radiation.group_bounds_eV too large: " + std::to_string(bounds_size) +
                        " exceeds int32-derived group limit");
    }
    radiation.groups = static_cast<int>(bounds_size - 1U);
  }

  if (has_key(kwargs, "volume_source_rate")) {
    radiation.volume_source_rate =
        numeric_as_double(kwargs["volume_source_rate"], "Radiation.volume_source_rate");
  }
  if (has_key(kwargs, "volume_source_x_max")) {
    radiation.volume_source_x_max = numeric_as_double(kwargs["volume_source_x_max"],
                                                      "Radiation.volume_source_x_max");
  }

  if (has_key(kwargs, "imc")) {
    const py::handle imc_obj = kwargs["imc"];
    if (!py::isinstance<py::dict>(imc_obj)) {
      throw_value_type_error("Radiation.imc", "dict", imc_obj);
    }
    const py::dict imc = py::reinterpret_borrow<py::dict>(imc_obj);
    enforce_known_keys(imc, "Radiation.imc",
                       {"enabled",
                        "alpha", "f_max", "corrected_fleck", "particles_per_cell_group",
                        "implicit_capture",
                        "cutoff_fraction", "inelastic_scatter", "weight_cutoff",
                        "roulette_survival", "weight_split", "max_split",
                        "linearized_planck", "source_tilting", "source_localization",
                        "sloc_ema_beta", "sloc_sigma_floor", "sloc_sigma_cap",
                        "sloc_tau_ref",
                        "spectral_bias_eta",
                        "opacity_predictor", "two_stage",
                        "difference",
                        "net_e_source_smoothing",
                        "conservative_smoother",
                        "particle_budget",
                        "census_comb", "rad_lite_mesh"});
    if (has_key(imc, "enabled")) {
      radiation.imc.enabled = strict_bool(imc["enabled"], "Radiation.imc.enabled");
    }
    if (has_key(imc, "alpha")) {
      radiation.imc.alpha = numeric_as_double(imc["alpha"], "Radiation.imc.alpha");
      if (!(radiation.imc.alpha > 0.0)) {
        throw ValueError("Radiation.imc.alpha must be > 0");
      }
    }
    if (has_key(imc, "f_max")) {
      radiation.imc.f_max = numeric_as_double(imc["f_max"], "Radiation.imc.f_max");
    }
    if (has_key(imc, "corrected_fleck")) {
      radiation.imc.corrected_fleck =
          strict_bool(imc["corrected_fleck"], "Radiation.imc.corrected_fleck");
    }
    if (has_key(imc, "particles_per_cell_group")) {
      radiation.imc.particles_per_cell_group = strict_int32(
          imc["particles_per_cell_group"],
          "Radiation.imc.particles_per_cell_group");
    }
    if (has_key(imc, "implicit_capture")) {
      radiation.imc.implicit_capture =
          strict_bool(imc["implicit_capture"], "Radiation.imc.implicit_capture");
    }
    if (has_key(imc, "cutoff_fraction")) {
      radiation.imc.cutoff_fraction = numeric_as_double(
          imc["cutoff_fraction"], "Radiation.imc.cutoff_fraction");
    }
    if (has_key(imc, "inelastic_scatter")) {
      radiation.imc.inelastic_scatter =
          strict_bool(imc["inelastic_scatter"], "Radiation.imc.inelastic_scatter");
    }
    if (has_key(imc, "weight_cutoff")) {
      radiation.imc.weight_cutoff =
          numeric_as_double(imc["weight_cutoff"], "Radiation.imc.weight_cutoff");
    }
    if (has_key(imc, "roulette_survival")) {
      radiation.imc.roulette_survival = numeric_as_double(
          imc["roulette_survival"], "Radiation.imc.roulette_survival");
    }
    if (has_key(imc, "weight_split")) {
      radiation.imc.weight_split =
          numeric_as_double(imc["weight_split"], "Radiation.imc.weight_split");
    }
    if (has_key(imc, "max_split")) {
      radiation.imc.max_split =
          strict_int32(imc["max_split"], "Radiation.imc.max_split");
    }
    if (has_key(imc, "linearized_planck")) {
      radiation.imc.linearized_planck =
          strict_bool(imc["linearized_planck"], "Radiation.imc.linearized_planck");
    }
    if (has_key(imc, "source_tilting")) {
      radiation.imc.source_tilting =
          strict_bool(imc["source_tilting"], "Radiation.imc.source_tilting");
    }
    if (has_key(imc, "source_localization")) {
      radiation.imc.source_localization = strict_bool(
          imc["source_localization"], "Radiation.imc.source_localization");
    }
    if (has_key(imc, "sloc_ema_beta")) {
      radiation.imc.sloc_ema_beta = numeric_as_double(
          imc["sloc_ema_beta"], "Radiation.imc.sloc_ema_beta");
    }
    if (has_key(imc, "sloc_sigma_floor")) {
      radiation.imc.sloc_sigma_floor = numeric_as_double(
          imc["sloc_sigma_floor"], "Radiation.imc.sloc_sigma_floor");
    }
    if (has_key(imc, "sloc_sigma_cap")) {
      radiation.imc.sloc_sigma_cap = numeric_as_double(
          imc["sloc_sigma_cap"], "Radiation.imc.sloc_sigma_cap");
    }
    if (has_key(imc, "sloc_tau_ref")) {
      radiation.imc.sloc_tau_ref = numeric_as_double(
          imc["sloc_tau_ref"], "Radiation.imc.sloc_tau_ref");
    }
    if (has_key(imc, "spectral_bias_eta")) {
      radiation.imc.spectral_bias_eta = numeric_as_double(
          imc["spectral_bias_eta"], "Radiation.imc.spectral_bias_eta");
    }
    if (has_key(imc, "opacity_predictor")) {
      radiation.imc.opacity_predictor =
          strict_bool(imc["opacity_predictor"], "Radiation.imc.opacity_predictor");
    }
    if (has_key(imc, "two_stage")) {
      radiation.imc.two_stage =
          strict_bool(imc["two_stage"], "Radiation.imc.two_stage");
    }
    if (has_key(imc, "difference")) {
      const py::handle difference_obj = imc["difference"];
      if (!py::isinstance<py::dict>(difference_obj)) {
        throw_value_type_error("Radiation.imc.difference", "dict", difference_obj);
      }
      const py::dict difference = py::reinterpret_borrow<py::dict>(difference_obj);
      enforce_known_keys(difference, "Radiation.imc.difference",
                         {"enabled", "W_max", "tau0", "chi0", "face_transport"});
      if (has_key(difference, "enabled")) {
        radiation.imc.difference.enabled =
            strict_bool(difference["enabled"], "Radiation.imc.difference.enabled");
      }
      if (has_key(difference, "W_max")) {
        radiation.imc.difference.W_max =
            numeric_as_double(difference["W_max"], "Radiation.imc.difference.W_max");
      }
      if (has_key(difference, "tau0")) {
        radiation.imc.difference.tau0 =
            numeric_as_double(difference["tau0"], "Radiation.imc.difference.tau0");
      }
      if (has_key(difference, "chi0")) {
        radiation.imc.difference.chi0 =
            numeric_as_double(difference["chi0"], "Radiation.imc.difference.chi0");
      }
      if (has_key(difference, "face_transport")) {
        radiation.imc.difference.face_transport = strict_bool(
            difference["face_transport"], "Radiation.imc.difference.face_transport");
      }
    }
    if (has_key(imc, "net_e_source_smoothing")) {
      const py::handle smoothing_obj = imc["net_e_source_smoothing"];
      if (!py::isinstance<py::dict>(smoothing_obj)) {
        throw_value_type_error("Radiation.imc.net_e_source_smoothing", "dict",
                               smoothing_obj);
      }
      const py::dict smoothing = py::reinterpret_borrow<py::dict>(smoothing_obj);
      enforce_known_keys(smoothing, "Radiation.imc.net_e_source_smoothing",
                         {"enabled", "alpha", "tau_threshold", "passes",
                          "grad_Te_scale", "grad_rho_scale", "gradient_adaptive"});
      if (has_key(smoothing, "enabled")) {
        radiation.imc.net_e_source_smoothing.enabled =
            strict_bool(smoothing["enabled"],
                        "Radiation.imc.net_e_source_smoothing.enabled");
      }
      if (has_key(smoothing, "alpha")) {
        radiation.imc.net_e_source_smoothing.alpha =
            numeric_as_double(smoothing["alpha"],
                              "Radiation.imc.net_e_source_smoothing.alpha");
      }
      if (has_key(smoothing, "tau_threshold")) {
        radiation.imc.net_e_source_smoothing.tau_threshold =
            numeric_as_double(
                smoothing["tau_threshold"],
                "Radiation.imc.net_e_source_smoothing.tau_threshold");
      }
      if (has_key(smoothing, "passes")) {
        radiation.imc.net_e_source_smoothing.passes =
            strict_int32(smoothing["passes"],
                         "Radiation.imc.net_e_source_smoothing.passes");
      }
      if (has_key(smoothing, "grad_Te_scale")) {
        radiation.imc.net_e_source_smoothing.grad_Te_scale =
            numeric_as_double(
                smoothing["grad_Te_scale"],
                "Radiation.imc.net_e_source_smoothing.grad_Te_scale");
      }
      if (has_key(smoothing, "grad_rho_scale")) {
        radiation.imc.net_e_source_smoothing.grad_rho_scale =
            numeric_as_double(
                smoothing["grad_rho_scale"],
                "Radiation.imc.net_e_source_smoothing.grad_rho_scale");
      }
      if (has_key(smoothing, "gradient_adaptive")) {
        radiation.imc.net_e_source_smoothing.gradient_adaptive =
            strict_bool(
                smoothing["gradient_adaptive"],
                "Radiation.imc.net_e_source_smoothing.gradient_adaptive");
      }
    }
    if (has_key(imc, "conservative_smoother")) {
      const py::handle smoother_obj = imc["conservative_smoother"];
      if (!py::isinstance<py::dict>(smoother_obj)) {
        throw_value_type_error("Radiation.imc.conservative_smoother", "dict",
                               smoother_obj);
      }
      const py::dict smoother = py::reinterpret_borrow<py::dict>(smoother_obj);
      enforce_known_keys(smoother, "Radiation.imc.conservative_smoother",
                         {"enabled", "passes", "alpha"});
      if (has_key(smoother, "enabled")) {
        radiation.imc.conservative_smoother.enabled =
            strict_bool(smoother["enabled"],
                        "Radiation.imc.conservative_smoother.enabled");
      }
      if (has_key(smoother, "passes")) {
        radiation.imc.conservative_smoother.passes =
            strict_int32(smoother["passes"],
                         "Radiation.imc.conservative_smoother.passes");
      }
      if (has_key(smoother, "alpha")) {
        radiation.imc.conservative_smoother.alpha =
            numeric_as_double(smoother["alpha"],
                              "Radiation.imc.conservative_smoother.alpha");
      }
    }
    if (has_key(imc, "particle_budget")) {
      radiation.imc.particle_budget = strict_int32(
          imc["particle_budget"], "Radiation.imc.particle_budget");
    }
    if (has_key(imc, "census_comb")) {
      const py::handle census_comb_obj = imc["census_comb"];
      if (!py::isinstance<py::dict>(census_comb_obj)) {
        throw_value_type_error("Radiation.imc.census_comb", "dict", census_comb_obj);
      }
      const py::dict census_comb = py::reinterpret_borrow<py::dict>(census_comb_obj);
      enforce_known_keys(census_comb, "Radiation.imc.census_comb",
                         {"enabled", "max_particles", "min_per_bin", "trigger_ratio",
                          "target_fraction", "mode_weight_imc", "mode_weight_ddmc",
                          "adaptive_trigger", "adaptive_util_start", "adaptive_util_end",
                          "trigger_ratio_floor", "trigger_hysteresis",
                          "ess_floor_enabled", "ess_min_tier0",
                          "ess_min_tier1", "max_split_factor"});
      if (has_key(census_comb, "enabled")) {
        radiation.imc.census_comb.enabled =
            strict_bool(census_comb["enabled"], "Radiation.imc.census_comb.enabled");
      }
      if (has_key(census_comb, "max_particles")) {
        radiation.imc.census_comb.max_particles = strict_int32(
            census_comb["max_particles"], "Radiation.imc.census_comb.max_particles");
      }
      if (has_key(census_comb, "min_per_bin")) {
        radiation.imc.census_comb.min_per_bin = strict_int32(
            census_comb["min_per_bin"], "Radiation.imc.census_comb.min_per_bin");
      }
      if (has_key(census_comb, "trigger_ratio")) {
        radiation.imc.census_comb.trigger_ratio = numeric_as_double(
            census_comb["trigger_ratio"], "Radiation.imc.census_comb.trigger_ratio");
      }
      if (has_key(census_comb, "target_fraction")) {
        radiation.imc.census_comb.target_fraction = numeric_as_double(
            census_comb["target_fraction"], "Radiation.imc.census_comb.target_fraction");
      }
      if (has_key(census_comb, "mode_weight_imc")) {
        radiation.imc.census_comb.mode_weight_imc = numeric_as_double(
            census_comb["mode_weight_imc"], "Radiation.imc.census_comb.mode_weight_imc");
      }
      if (has_key(census_comb, "mode_weight_ddmc")) {
        radiation.imc.census_comb.mode_weight_ddmc = numeric_as_double(
            census_comb["mode_weight_ddmc"], "Radiation.imc.census_comb.mode_weight_ddmc");
      }
      if (has_key(census_comb, "adaptive_trigger")) {
        radiation.imc.census_comb.adaptive_trigger = strict_bool(
            census_comb["adaptive_trigger"], "Radiation.imc.census_comb.adaptive_trigger");
      }
      if (has_key(census_comb, "adaptive_util_start")) {
        radiation.imc.census_comb.adaptive_util_start = numeric_as_double(
            census_comb["adaptive_util_start"], "Radiation.imc.census_comb.adaptive_util_start");
      }
      if (has_key(census_comb, "adaptive_util_end")) {
        radiation.imc.census_comb.adaptive_util_end = numeric_as_double(
            census_comb["adaptive_util_end"], "Radiation.imc.census_comb.adaptive_util_end");
      }
      if (has_key(census_comb, "trigger_ratio_floor")) {
        radiation.imc.census_comb.trigger_ratio_floor = numeric_as_double(
            census_comb["trigger_ratio_floor"], "Radiation.imc.census_comb.trigger_ratio_floor");
      }
      if (has_key(census_comb, "trigger_hysteresis")) {
        radiation.imc.census_comb.trigger_hysteresis = numeric_as_double(
            census_comb["trigger_hysteresis"], "Radiation.imc.census_comb.trigger_hysteresis");
      }
      if (has_key(census_comb, "ess_floor_enabled")) {
        radiation.imc.census_comb.ess_floor_enabled = strict_bool(
            census_comb["ess_floor_enabled"], "Radiation.imc.census_comb.ess_floor_enabled");
      }
      if (has_key(census_comb, "ess_min_tier0")) {
        radiation.imc.census_comb.ess_min_tier0 = numeric_as_double(
            census_comb["ess_min_tier0"], "Radiation.imc.census_comb.ess_min_tier0");
      }
      if (has_key(census_comb, "ess_min_tier1")) {
        radiation.imc.census_comb.ess_min_tier1 = numeric_as_double(
            census_comb["ess_min_tier1"], "Radiation.imc.census_comb.ess_min_tier1");
      }
      if (has_key(census_comb, "max_split_factor")) {
        radiation.imc.census_comb.max_split_factor = strict_int32(
            census_comb["max_split_factor"], "Radiation.imc.census_comb.max_split_factor");
      }
    }
    if (has_key(imc, "rad_lite_mesh")) {
      const py::handle rlm_obj = imc["rad_lite_mesh"];
      if (!py::isinstance<py::dict>(rlm_obj)) {
        throw_value_type_error("Radiation.imc.rad_lite_mesh", "dict", rlm_obj);
      }
      const py::dict rlm = py::reinterpret_borrow<py::dict>(rlm_obj);
      enforce_known_keys(rlm, "Radiation.imc.rad_lite_mesh",
                         {"enabled", "sigma_ratio_max", "nlte_auto"});
      if (has_key(rlm, "enabled")) {
        radiation.imc.rad_lite_mesh.enabled =
            strict_bool(rlm["enabled"], "Radiation.imc.rad_lite_mesh.enabled");
      }
      if (has_key(rlm, "sigma_ratio_max")) {
        radiation.imc.rad_lite_mesh.sigma_ratio_max = numeric_as_double(
            rlm["sigma_ratio_max"], "Radiation.imc.rad_lite_mesh.sigma_ratio_max");
        if (!(radiation.imc.rad_lite_mesh.sigma_ratio_max > 1.0)) {
          throw ValueError("Radiation.imc.rad_lite_mesh.sigma_ratio_max must be > 1.0");
        }
      }
      if (has_key(rlm, "nlte_auto")) {
        radiation.imc.rad_lite_mesh.nlte_auto =
            strict_bool(rlm["nlte_auto"], "Radiation.imc.rad_lite_mesh.nlte_auto");
      }
    }
  }

  if (has_key(kwargs, "ddmc")) {
    const py::handle ddmc_obj = kwargs["ddmc"];
    if (!py::isinstance<py::dict>(ddmc_obj)) {
      throw_value_type_error("Radiation.ddmc", "dict", ddmc_obj);
    }
    const py::dict ddmc = py::reinterpret_borrow<py::dict>(ddmc_obj);
    enforce_known_keys(ddmc, "Radiation.ddmc",
                       {"enabled", "implicit_diffusion", "tau_ddmc", "tau_rw", "omega_ddmc", "leak_stencil",
                        "tau_ddmc_off", "omega_ddmc_off", "mode_hold", "rate_max",
                        "interface_method", "emissivity_preserving",
                        "interface_exit_distribution", "rz_face_r_weight",
                        "face_opacity_temperature", "m_matrix_check"});
    if (has_key(ddmc, "enabled")) {
      radiation.ddmc.enabled = strict_bool(ddmc["enabled"], "Radiation.ddmc.enabled");
    }
    if (has_key(ddmc, "implicit_diffusion")) {
      radiation.ddmc.implicit_diffusion = strict_bool(
          ddmc["implicit_diffusion"], "Radiation.ddmc.implicit_diffusion");
    }
    if (has_key(ddmc, "tau_ddmc")) {
      radiation.ddmc.tau_ddmc =
          numeric_as_double(ddmc["tau_ddmc"], "Radiation.ddmc.tau_ddmc");
    }
    if (has_key(ddmc, "tau_rw")) {
      radiation.ddmc.tau_rw =
          numeric_as_double(ddmc["tau_rw"], "Radiation.ddmc.tau_rw");
    }
    if (has_key(ddmc, "omega_ddmc")) {
      radiation.ddmc.omega_ddmc =
          numeric_as_double(ddmc["omega_ddmc"], "Radiation.ddmc.omega_ddmc");
    }
    if (has_key(ddmc, "tau_ddmc_off")) {
      radiation.ddmc.tau_ddmc_off =
          numeric_as_double(ddmc["tau_ddmc_off"], "Radiation.ddmc.tau_ddmc_off");
    }
    if (has_key(ddmc, "omega_ddmc_off")) {
      radiation.ddmc.omega_ddmc_off = numeric_as_double(
          ddmc["omega_ddmc_off"], "Radiation.ddmc.omega_ddmc_off");
    }
    if (has_key(ddmc, "mode_hold")) {
      radiation.ddmc.mode_hold = strict_int32(ddmc["mode_hold"], "Radiation.ddmc.mode_hold");
    }
    if (has_key(ddmc, "rate_max")) {
      radiation.ddmc.rate_max =
          numeric_as_double(ddmc["rate_max"], "Radiation.ddmc.rate_max");
    }
    if (has_key(ddmc, "leak_stencil")) {
      radiation.ddmc.leak_stencil =
          strict_string(ddmc["leak_stencil"], "Radiation.ddmc.leak_stencil");
    }
    if (has_key(ddmc, "interface_method")) {
      radiation.ddmc.interface_method =
          strict_string(ddmc["interface_method"],
                        "Radiation.ddmc.interface_method");
    }
    if (has_key(ddmc, "emissivity_preserving")) {
      radiation.ddmc.emissivity_preserving = strict_bool(
          ddmc["emissivity_preserving"], "Radiation.ddmc.emissivity_preserving");
    }
    if (has_key(ddmc, "interface_exit_distribution")) {
      radiation.ddmc.interface_exit_distribution =
          strict_string(ddmc["interface_exit_distribution"],
                        "Radiation.ddmc.interface_exit_distribution");
    }
    if (has_key(ddmc, "rz_face_r_weight")) {
      radiation.ddmc.rz_face_r_weight =
          strict_bool(ddmc["rz_face_r_weight"], "Radiation.ddmc.rz_face_r_weight");
    }
    if (has_key(ddmc, "face_opacity_temperature")) {
      radiation.ddmc.face_opacity_temperature =
          strict_string(ddmc["face_opacity_temperature"],
                        "Radiation.ddmc.face_opacity_temperature");
    }
    if (has_key(ddmc, "m_matrix_check")) {
      radiation.ddmc.m_matrix_check = strict_bool(ddmc["m_matrix_check"],
                                                  "Radiation.ddmc.m_matrix_check");
    }
  }

  if (has_key(kwargs, "diffusion")) {
    const py::handle diffusion_obj = kwargs["diffusion"];
    if (!py::isinstance<py::dict>(diffusion_obj)) {
      throw_value_type_error("Radiation.diffusion", "dict", diffusion_obj);
    }
    const py::dict diffusion = py::reinterpret_borrow<py::dict>(diffusion_obj);
    enforce_known_keys(diffusion, "Radiation.diffusion",
                       {"enabled", "tau_on", "tau_off",
                        "reduced_flux_on", "reduced_flux_off",
                        "mode_hold", "rate_max",
                        "mode_update_interval", "min_diffusion_island_cells",
                        "imc_guard_cells",
                        "sts_max_stages", "sts_damping", "sts_subcycle_eta",
                        "interface_particles_per_face_group",
                        "exit_particles_per_cell_group",
                        "lte_entry_initialization",
                        "lte_entry_energy_fraction_cap"});
    if (has_key(diffusion, "enabled")) {
      radiation.diffusion.enabled =
          strict_bool(diffusion["enabled"], "Radiation.diffusion.enabled");
    }
    if (has_key(diffusion, "tau_on")) {
      radiation.diffusion.tau_on =
          numeric_as_double(diffusion["tau_on"], "Radiation.diffusion.tau_on");
    }
    if (has_key(diffusion, "tau_off")) {
      radiation.diffusion.tau_off =
          numeric_as_double(diffusion["tau_off"], "Radiation.diffusion.tau_off");
    }
    if (has_key(diffusion, "reduced_flux_on")) {
      radiation.diffusion.reduced_flux_on = numeric_as_double(
          diffusion["reduced_flux_on"], "Radiation.diffusion.reduced_flux_on");
    }
    if (has_key(diffusion, "reduced_flux_off")) {
      radiation.diffusion.reduced_flux_off = numeric_as_double(
          diffusion["reduced_flux_off"], "Radiation.diffusion.reduced_flux_off");
    }
    if (has_key(diffusion, "mode_hold")) {
      radiation.diffusion.mode_hold =
          strict_int32(diffusion["mode_hold"], "Radiation.diffusion.mode_hold");
    }
    if (has_key(diffusion, "rate_max")) {
      radiation.diffusion.rate_max =
          numeric_as_double(diffusion["rate_max"], "Radiation.diffusion.rate_max");
    }
    if (has_key(diffusion, "mode_update_interval")) {
      radiation.diffusion.mode_update_interval = strict_int32(
          diffusion["mode_update_interval"],
          "Radiation.diffusion.mode_update_interval");
    }
    if (has_key(diffusion, "min_diffusion_island_cells")) {
      radiation.diffusion.min_diffusion_island_cells = strict_int32(
          diffusion["min_diffusion_island_cells"],
          "Radiation.diffusion.min_diffusion_island_cells");
    }
    if (has_key(diffusion, "imc_guard_cells")) {
      radiation.diffusion.imc_guard_cells = strict_int32(
          diffusion["imc_guard_cells"], "Radiation.diffusion.imc_guard_cells");
    }
    if (has_key(diffusion, "sts_max_stages")) {
      radiation.diffusion.sts_max_stages = strict_int32(
          diffusion["sts_max_stages"], "Radiation.diffusion.sts_max_stages");
    }
    if (has_key(diffusion, "sts_damping")) {
      radiation.diffusion.sts_damping = numeric_as_double(
          diffusion["sts_damping"], "Radiation.diffusion.sts_damping");
    }
    if (has_key(diffusion, "sts_subcycle_eta")) {
      radiation.diffusion.sts_subcycle_eta = numeric_as_double(
          diffusion["sts_subcycle_eta"], "Radiation.diffusion.sts_subcycle_eta");
    }
    if (has_key(diffusion, "interface_particles_per_face_group")) {
      radiation.diffusion.interface_particles_per_face_group = strict_int32(
          diffusion["interface_particles_per_face_group"],
          "Radiation.diffusion.interface_particles_per_face_group");
    }
    if (has_key(diffusion, "exit_particles_per_cell_group")) {
      radiation.diffusion.exit_particles_per_cell_group = strict_int32(
          diffusion["exit_particles_per_cell_group"],
          "Radiation.diffusion.exit_particles_per_cell_group");
    }
    if (has_key(diffusion, "lte_entry_initialization")) {
      radiation.diffusion.lte_entry_initialization = strict_bool(
          diffusion["lte_entry_initialization"],
          "Radiation.diffusion.lte_entry_initialization");
    }
    if (has_key(diffusion, "lte_entry_energy_fraction_cap")) {
      radiation.diffusion.lte_entry_energy_fraction_cap = numeric_as_double(
          diffusion["lte_entry_energy_fraction_cap"],
          "Radiation.diffusion.lte_entry_energy_fraction_cap");
    }
  }

  if (has_key(kwargs, "multigroup_diffusion")) {
    const py::handle fld_obj = kwargs["multigroup_diffusion"];
    if (!py::isinstance<py::dict>(fld_obj)) {
      throw_value_type_error("Radiation.multigroup_diffusion", "dict", fld_obj);
    }
    const py::dict fld = py::reinterpret_borrow<py::dict>(fld_obj);
    enforce_known_keys(fld,
                       "Radiation.multigroup_diffusion",
                       {"flux_limiter",
                        "max_outer_iterations",
                        "outer_tol",
                        "fleck_mode",
                        "fleck_cv_source",
                        "fleck_beta",
                        "fleck_form",
                        "source_integrator",
                        "hydro_coupling",
                        "state_supply_boundary_policy",
                        "diagnostic_radial_fourier_substage_enabled",
                        "cg_inner_tol",
                        "cg_tol_norm",
                        "outer_accel",
                        "anderson_m",
                        "anderson_beta",
                        "cg_max_iter",
                        "cap_exit_policy",
                        "linear_solver_1d",
                        "linear_solver_2d",
                        "rgmg_smoother_omega",
                        "amgx_config",
                        "z_boundary",
                        "opacity_floor",
                        "opacity_cap",
                        "marshak",
                        "boundary"});
    auto& fld_cfg = radiation.multigroup_diffusion;
    if (has_key(fld, "flux_limiter")) {
      fld_cfg.flux_limiter = strict_string(
          fld["flux_limiter"], "Radiation.multigroup_diffusion.flux_limiter");
    }
    if (has_key(fld, "max_outer_iterations")) {
      fld_cfg.max_outer_iterations = strict_int32(
          fld["max_outer_iterations"],
          "Radiation.multigroup_diffusion.max_outer_iterations");
    }
    if (has_key(fld, "fleck_mode")) {
      fld_cfg.fleck_mode =
          strict_string(fld["fleck_mode"],
                        "Radiation.multigroup_diffusion.fleck_mode");
      if (fld_cfg.fleck_mode != "fleck_cummings" &&
          fld_cfg.fleck_mode != "afi") {
        throw ConfigError(
            "Radiation.multigroup_diffusion.fleck_mode must be"
            " \"fleck_cummings\" or \"afi\"");
      }
    }
    if (has_key(fld, "fleck_cv_source")) {
      fld_cfg.fleck_cv_source =
          strict_string(fld["fleck_cv_source"],
                        "Radiation.multigroup_diffusion.fleck_cv_source");
      if (fld_cfg.fleck_cv_source != "legacy" &&
          fld_cfg.fleck_cv_source != "table") {
        throw ConfigError(
            "Radiation.multigroup_diffusion.fleck_cv_source must be"
            " \"legacy\" or \"table\"");
      }
    }
    if (has_key(fld, "fleck_beta")) {
      fld_cfg.fleck_beta =
          strict_string(fld["fleck_beta"],
                        "Radiation.multigroup_diffusion.fleck_beta");
      if (fld_cfg.fleck_beta != "tangent" &&
          fld_cfg.fleck_beta != "secant" &&
          fld_cfg.fleck_beta != "guard") {
        throw ConfigError(
            "Radiation.multigroup_diffusion.fleck_beta must be"
            " \"tangent\", \"secant\", or \"guard\"");
      }
    }
    if (has_key(fld, "fleck_form")) {
      fld_cfg.fleck_form =
          strict_string(fld["fleck_form"],
                        "Radiation.multigroup_diffusion.fleck_form");
      if (fld_cfg.fleck_form != "be" &&
          fld_cfg.fleck_form != "exp_phi1") {
        throw ConfigError(
            "Radiation.multigroup_diffusion.fleck_form must be"
            " \"be\" or \"exp_phi1\"");
      }
    }
    if (has_key(fld, "source_integrator")) {
      fld_cfg.source_integrator =
          strict_string(fld["source_integrator"],
                        "Radiation.multigroup_diffusion.source_integrator");
      if (fld_cfg.source_integrator != "fleck" &&
          fld_cfg.source_integrator != "exp_rosenbrock") {
        throw ConfigError(
            "Radiation.multigroup_diffusion.source_integrator must be"
            " \"fleck\" or \"exp_rosenbrock\"");
      }
    }
    if (has_key(fld, "hydro_coupling")) {
      fld_cfg.hydro_coupling =
          strict_string(fld["hydro_coupling"],
                        "Radiation.multigroup_diffusion.hydro_coupling");
      if (fld_cfg.hydro_coupling != "none" &&
          fld_cfg.hydro_coupling != "gamma_r_43") {
        throw ConfigError(
            "Radiation.multigroup_diffusion.hydro_coupling must be"
            " \"none\" or \"gamma_r_43\"");
      }
    }
    if (has_key(fld, "outer_tol")) {
      fld_cfg.outer_tol =
          numeric_as_double(fld["outer_tol"], "Radiation.multigroup_diffusion.outer_tol");
    }
    if (has_key(fld, "state_supply_boundary_policy")) {
      fld_cfg.state_supply_boundary_policy = strict_string(
          fld["state_supply_boundary_policy"],
          "Radiation.multigroup_diffusion.state_supply_boundary_policy");
    }
    if (has_key(fld, "diagnostic_radial_fourier_substage_enabled")) {
      fld_cfg.diagnostic_radial_fourier_substage_enabled = strict_bool(
          fld["diagnostic_radial_fourier_substage_enabled"],
          "Radiation.multigroup_diffusion.diagnostic_radial_fourier_substage_enabled");
    }
    if (has_key(fld, "cg_inner_tol")) {
      fld_cfg.cg_inner_tol = numeric_as_double(
          fld["cg_inner_tol"], "Radiation.multigroup_diffusion.cg_inner_tol");
    }
    if (has_key(fld, "cg_tol_norm")) {
      fld_cfg.cg_tol_norm = strict_string(
          fld["cg_tol_norm"], "Radiation.multigroup_diffusion.cg_tol_norm");
    }
    if (has_key(fld, "outer_accel")) {
      fld_cfg.outer_accel = strict_string(
          fld["outer_accel"], "Radiation.multigroup_diffusion.outer_accel");
    }
    if (has_key(fld, "anderson_m")) {
      fld_cfg.anderson_m = strict_int32(
          fld["anderson_m"], "Radiation.multigroup_diffusion.anderson_m");
    }
    if (has_key(fld, "anderson_beta")) {
      fld_cfg.anderson_beta = numeric_as_double(
          fld["anderson_beta"], "Radiation.multigroup_diffusion.anderson_beta");
    }
    if (has_key(fld, "cg_max_iter")) {
      fld_cfg.cg_max_iter = strict_int32(
          fld["cg_max_iter"], "Radiation.multigroup_diffusion.cg_max_iter");
    }
    if (has_key(fld, "cap_exit_policy")) {
      fld_cfg.cap_exit_policy = strict_string(
          fld["cap_exit_policy"],
          "Radiation.multigroup_diffusion.cap_exit_policy");
    }
    if (has_key(fld, "linear_solver_1d")) {
      fld_cfg.linear_solver_1d = strict_string(
          fld["linear_solver_1d"],
          "Radiation.multigroup_diffusion.linear_solver_1d");
    }
    if (has_key(fld, "linear_solver_2d")) {
      fld_cfg.linear_solver_2d = strict_string(
          fld["linear_solver_2d"],
          "Radiation.multigroup_diffusion.linear_solver_2d");
      fld_cfg.linear_solver_2d_explicit = true;
      fld_cfg.linear_solver_2d_requested = fld_cfg.linear_solver_2d;
    }
    if (has_key(fld, "rgmg_smoother_omega")) {
      fld_cfg.rgmg_smoother_omega =
          numeric_as_double(fld["rgmg_smoother_omega"],
                            "Radiation.multigroup_diffusion.rgmg_smoother_omega");
    }
    if (has_key(fld, "amgx_config")) {
      const py::handle amgx_obj = fld["amgx_config"];
      if (!py::isinstance<py::dict>(amgx_obj)) {
        throw_value_type_error(
            "Radiation.multigroup_diffusion.amgx_config", "dict", amgx_obj);
      }
      const py::dict amgx = py::reinterpret_borrow<py::dict>(amgx_obj);
      enforce_known_keys(amgx,
                         "Radiation.multigroup_diffusion.amgx_config",
                         {"preset"});
      if (has_key(amgx, "preset")) {
        fld_cfg.amgx_config.preset = strict_string(
            amgx["preset"],
            "Radiation.multigroup_diffusion.amgx_config.preset");
      }
    }
    if (has_key(fld, "z_boundary")) {
      fld_cfg.z_boundary = strict_string(
          fld["z_boundary"], "Radiation.multigroup_diffusion.z_boundary");
      fld_cfg.boundary.z = fld_cfg.z_boundary;
      fld_cfg.boundary.z_bottom = fld_cfg.z_boundary;
      fld_cfg.boundary.z_top = fld_cfg.z_boundary;
    }
    if (has_key(fld, "opacity_floor")) {
      fld_cfg.opacity_floor = numeric_as_double(
          fld["opacity_floor"], "Radiation.multigroup_diffusion.opacity_floor");
    }
    if (has_key(fld, "opacity_cap")) {
      fld_cfg.opacity_cap = numeric_as_double(
          fld["opacity_cap"], "Radiation.multigroup_diffusion.opacity_cap");
    }
    if (has_key(fld, "marshak")) {
      const py::handle marshak_obj = fld["marshak"];
      if (!py::isinstance<py::dict>(marshak_obj)) {
        throw_value_type_error(
            "Radiation.multigroup_diffusion.marshak", "dict", marshak_obj);
      }
      const py::dict marshak = py::reinterpret_borrow<py::dict>(marshak_obj);
      enforce_known_keys(marshak,
                         "Radiation.multigroup_diffusion.marshak",
                         {"flux_erg_per_cm2_s", "flux_pulse_duration_s"});
      if (has_key(marshak, "flux_erg_per_cm2_s")) {
        fld_cfg.marshak.flux_erg_per_cm2_s = numeric_as_double(
            marshak["flux_erg_per_cm2_s"],
            "Radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s");
      }
      if (has_key(marshak, "flux_pulse_duration_s")) {
        fld_cfg.marshak.flux_pulse_duration_s = numeric_as_double(
            marshak["flux_pulse_duration_s"],
            "Radiation.multigroup_diffusion.marshak.flux_pulse_duration_s");
      }
    }
    if (has_key(fld, "boundary")) {
      const py::handle boundary_obj = fld["boundary"];
      if (!py::isinstance<py::dict>(boundary_obj)) {
        throw_value_type_error(
            "Radiation.multigroup_diffusion.boundary", "dict", boundary_obj);
      }
      const py::dict boundary = py::reinterpret_borrow<py::dict>(boundary_obj);
      enforce_known_keys(boundary,
                         "Radiation.multigroup_diffusion.boundary",
                         {"inner_r", "outer_r", "z", "z_bottom", "z_top"});
      if (has_key(boundary, "inner_r")) {
        fld_cfg.boundary.inner_r = strict_string(
            boundary["inner_r"],
            "Radiation.multigroup_diffusion.boundary.inner_r");
      }
      if (has_key(boundary, "outer_r")) {
        fld_cfg.boundary.outer_r = strict_string(
            boundary["outer_r"],
            "Radiation.multigroup_diffusion.boundary.outer_r");
      }
      if (has_key(boundary, "z")) {
        fld_cfg.boundary.z = strict_string(
            boundary["z"],
            "Radiation.multigroup_diffusion.boundary.z");
        if (has_key(fld, "z_boundary") &&
            fld_cfg.boundary.z != fld_cfg.z_boundary) {
          throw ConfigError(
              "Radiation.multigroup_diffusion.z_boundary and boundary.z must match when both are specified");
        }
        fld_cfg.z_boundary = fld_cfg.boundary.z;
        fld_cfg.boundary.z_bottom = fld_cfg.boundary.z;
        fld_cfg.boundary.z_top = fld_cfg.boundary.z;
      }
      if (has_key(boundary, "z_bottom")) {
        fld_cfg.boundary.z_bottom = strict_string(
            boundary["z_bottom"],
            "Radiation.multigroup_diffusion.boundary.z_bottom");
      }
      if (has_key(boundary, "z_top")) {
        fld_cfg.boundary.z_top = strict_string(
            boundary["z_top"],
            "Radiation.multigroup_diffusion.boundary.z_top");
      }
    }
    if (fld_cfg.source_integrator == "exp_rosenbrock") {
      if (radiation.groups > 96) {
        throw ConfigError(
            "Radiation.multigroup_diffusion.source_integrator="
            "\"exp_rosenbrock\" supports at most 96 groups");
      }
      if (fld_cfg.fleck_mode == "afi") {
        throw ConfigError(
            "source_integrator=\"exp_rosenbrock\" is incompatible with"
            " fleck_mode=\"afi\"");
      }
      if (fld_cfg.fleck_form == "exp_phi1") {
        throw ConfigError(
            "source_integrator=\"exp_rosenbrock\" does not consume the Fleck"
            " factor; fleck_form=\"exp_phi1\" would be a dead knob — use one"
            " or the other");
      }
      if (fld_cfg.fleck_beta != "tangent") {
        throw ConfigError(
            "source_integrator=\"exp_rosenbrock\" v1 uses the tangent beta"
            " only (the secant/guard predictor lives in the Fleck kernel;"
            " composing it here would duplicate certified logic)");
      }
    }
  }

  if (has_key(kwargs, "sn_transport")) {
    const py::handle sn_obj = kwargs["sn_transport"];
    if (!py::isinstance<py::dict>(sn_obj)) {
      throw_value_type_error("Radiation.sn_transport", "dict", sn_obj);
    }
    const py::dict sn = py::reinterpret_borrow<py::dict>(sn_obj);
    enforce_known_keys(sn,
                       "Radiation.sn_transport",
                       {"n_angles",
                        "angular_quadrature",
                        "spatial_scheme",
                        "max_outer_iterations",
                        "max_inner_iterations",
                        "outer_tol",
                        "outer_tol_stagnation_factor",
                        "outer_tol_hydro_error_scale",
                        "inner_tol",
                        "inner_graph_unroll",
                        "dsa_enabled",
                        "z_boundary",
                        "diffusion_fallback_mode",
                        "tau_diffusion_on",
                        "tau_diffusion_off",
                        "opacity_floor",
                        "opacity_cap",
                        "timing_enabled",
                        "marshak",
                        // Deprecated 1D_SPH closure selectors: accepted for
                        // legacy namelists but ignored; production closure is fixed.
                        "local_implicit_source",
                        "coupling",
                        "e_star_source",
                        "streaming_limiter",
                        "ap_hybrid",
                        "boundary"});
    auto& sn_cfg = radiation.sn_transport;
    if (has_key(sn, "n_angles")) {
      sn_cfg.n_angles =
          strict_int32(sn["n_angles"], "Radiation.sn_transport.n_angles");
    }
    if (has_key(sn, "angular_quadrature")) {
      sn_cfg.angular_quadrature = strict_string(
          sn["angular_quadrature"], "Radiation.sn_transport.angular_quadrature");
      const std::string prefix = "level_symmetric_";
      if (sn_cfg.angular_quadrature.rfind(prefix, 0) == 0) {
        try {
          sn_cfg.n_angles = std::stoi(sn_cfg.angular_quadrature.substr(prefix.size()));
        } catch (const std::exception&) {
          throw ValueError(
              "Radiation.sn_transport.angular_quadrature has invalid level_symmetric suffix");
        }
      }
    }
    if (has_key(sn, "spatial_scheme")) {
      sn_cfg.spatial_scheme = strict_string(
          sn["spatial_scheme"], "Radiation.sn_transport.spatial_scheme");
    }
    if (has_key(sn, "max_outer_iterations")) {
      sn_cfg.max_outer_iterations = strict_int32(
          sn["max_outer_iterations"],
          "Radiation.sn_transport.max_outer_iterations");
    }
    if (has_key(sn, "max_inner_iterations")) {
      sn_cfg.max_inner_iterations = strict_int32(
          sn["max_inner_iterations"],
          "Radiation.sn_transport.max_inner_iterations");
    }
    if (has_key(sn, "outer_tol")) {
      sn_cfg.outer_tol =
          numeric_as_double(sn["outer_tol"], "Radiation.sn_transport.outer_tol");
    }
    if (has_key(sn, "outer_tol_stagnation_factor")) {
      sn_cfg.outer_tol_stagnation_factor =
          numeric_as_double(sn["outer_tol_stagnation_factor"],
                            "Radiation.sn_transport.outer_tol_stagnation_factor");
    }
    if (has_key(sn, "outer_tol_hydro_error_scale")) {
      sn_cfg.outer_tol_hydro_error_scale =
          numeric_as_double(sn["outer_tol_hydro_error_scale"],
                            "Radiation.sn_transport.outer_tol_hydro_error_scale");
    }
    if (has_key(sn, "inner_tol")) {
      sn_cfg.inner_tol =
          numeric_as_double(sn["inner_tol"], "Radiation.sn_transport.inner_tol");
    }
    if (has_key(sn, "inner_graph_unroll")) {
      sn_cfg.inner_graph_unroll =
          strict_int32(sn["inner_graph_unroll"],
                       "Radiation.sn_transport.inner_graph_unroll");
    }
    if (has_key(sn, "dsa_enabled")) {
      sn_cfg.dsa_enabled =
          strict_bool(sn["dsa_enabled"], "Radiation.sn_transport.dsa_enabled");
    }
    if (has_key(sn, "z_boundary")) {
      sn_cfg.z_boundary =
          strict_string(sn["z_boundary"], "Radiation.sn_transport.z_boundary");
      sn_cfg.boundary.z = sn_cfg.z_boundary;
      sn_cfg.boundary.z_bottom = sn_cfg.z_boundary;
      sn_cfg.boundary.z_top = sn_cfg.z_boundary;
    }
    if (has_key(sn, "diffusion_fallback_mode")) {
      sn_cfg.diffusion_fallback_mode = strict_string(
          sn["diffusion_fallback_mode"],
          "Radiation.sn_transport.diffusion_fallback_mode");
    }
    if (has_key(sn, "tau_diffusion_on")) {
      sn_cfg.tau_diffusion_on = numeric_as_double(
          sn["tau_diffusion_on"], "Radiation.sn_transport.tau_diffusion_on");
    }
    if (has_key(sn, "tau_diffusion_off")) {
      sn_cfg.tau_diffusion_off = numeric_as_double(
          sn["tau_diffusion_off"], "Radiation.sn_transport.tau_diffusion_off");
    }
    if (has_key(sn, "opacity_floor")) {
      sn_cfg.opacity_floor = numeric_as_double(
          sn["opacity_floor"], "Radiation.sn_transport.opacity_floor");
    }
    if (has_key(sn, "opacity_cap")) {
      sn_cfg.opacity_cap = numeric_as_double(
          sn["opacity_cap"], "Radiation.sn_transport.opacity_cap");
    }
    if (has_key(sn, "timing_enabled")) {
      sn_cfg.timing_enabled =
          strict_bool(sn["timing_enabled"], "Radiation.sn_transport.timing_enabled");
    }
    if (has_key(sn, "marshak")) {
      const py::handle marshak_obj = sn["marshak"];
      if (!py::isinstance<py::dict>(marshak_obj)) {
        throw_value_type_error(
            "Radiation.sn_transport.marshak", "dict", marshak_obj);
      }
      const py::dict marshak = py::reinterpret_borrow<py::dict>(marshak_obj);
      enforce_known_keys(marshak,
                         "Radiation.sn_transport.marshak",
                         {"flux_erg_per_cm2_s", "flux_pulse_duration_s"});
      if (has_key(marshak, "flux_erg_per_cm2_s")) {
        sn_cfg.marshak.flux_erg_per_cm2_s = numeric_as_double(
            marshak["flux_erg_per_cm2_s"],
            "Radiation.sn_transport.marshak.flux_erg_per_cm2_s");
      }
      if (has_key(marshak, "flux_pulse_duration_s")) {
        sn_cfg.marshak.flux_pulse_duration_s = numeric_as_double(
            marshak["flux_pulse_duration_s"],
            "Radiation.sn_transport.marshak.flux_pulse_duration_s");
      }
    }
    if (has_key(sn, "boundary")) {
      const py::handle boundary_obj = sn["boundary"];
      if (!py::isinstance<py::dict>(boundary_obj)) {
        throw_value_type_error(
            "Radiation.sn_transport.boundary", "dict", boundary_obj);
      }
      const py::dict boundary = py::reinterpret_borrow<py::dict>(boundary_obj);
      enforce_known_keys(boundary,
                         "Radiation.sn_transport.boundary",
                         {"inner_r", "outer_r", "z", "z_bottom", "z_top"});
      if (has_key(boundary, "inner_r")) {
        sn_cfg.boundary.inner_r = strict_string(
            boundary["inner_r"],
            "Radiation.sn_transport.boundary.inner_r");
      }
      if (has_key(boundary, "outer_r")) {
        sn_cfg.boundary.outer_r = strict_string(
            boundary["outer_r"],
            "Radiation.sn_transport.boundary.outer_r");
      }
      if (has_key(boundary, "z")) {
        sn_cfg.boundary.z = strict_string(
            boundary["z"],
            "Radiation.sn_transport.boundary.z");
        if (has_key(sn, "z_boundary") &&
            sn_cfg.boundary.z != sn_cfg.z_boundary) {
          throw ConfigError(
              "Radiation.sn_transport.z_boundary and boundary.z must match when both are specified");
        }
        sn_cfg.z_boundary = sn_cfg.boundary.z;
        sn_cfg.boundary.z_bottom = sn_cfg.boundary.z;
        sn_cfg.boundary.z_top = sn_cfg.boundary.z;
      }
      if (has_key(boundary, "z_bottom")) {
        sn_cfg.boundary.z_bottom = strict_string(
            boundary["z_bottom"],
            "Radiation.sn_transport.boundary.z_bottom");
      }
      if (has_key(boundary, "z_top")) {
        sn_cfg.boundary.z_top = strict_string(
            boundary["z_top"],
            "Radiation.sn_transport.boundary.z_top");
      }
    }
  }

  if (has_key(kwargs, "holo")) {
    const py::handle holo_obj = kwargs["holo"];
    if (!py::isinstance<py::dict>(holo_obj)) {
      throw_value_type_error("Radiation.holo", "dict", holo_obj);
    }
    const py::dict holo = py::reinterpret_borrow<py::dict>(holo_obj);
    const bool holo_has_tau_on = has_key(holo, "tau_on");
    const bool holo_has_tau_off = has_key(holo, "tau_off");
    enforce_known_keys(holo, "Radiation.holo",
                       {"enabled", "region", "material_group", "q_min", "q_max",
                        "coupling_tau", "guard_cells", "blend_cells",
                        "min_lo_cells",
                        "tau_on", "tau_off",
                        "reduced_flux_on", "reduced_flux_off",
                        "update_interval", "hold_on", "min_dwell_steps",
                        "min_island_cells", "core_margin_cells",
                        "solver", "closure", "closure_relax",
                        "closure_smooth_passes", "closure_smooth_alpha",
                        "consistency_alpha", "gamma_alpha", "boundary_flux", "p_rr_tally",
                        "sn_closure", "sn_n_angles",
                        "sn_material_coupling",
                        "residual_particles_per_cell_group"});
    if (has_key(holo, "enabled")) {
      radiation.holo.enabled = strict_bool(holo["enabled"], "Radiation.holo.enabled");
    }
    if (has_key(holo, "region")) {
      radiation.holo.region = strict_string(holo["region"], "Radiation.holo.region");
    }
    if (has_key(holo, "material_group")) {
      radiation.holo.material_group =
          strict_string(holo["material_group"], "Radiation.holo.material_group");
    }
    if (has_key(holo, "coupling_tau")) {
      radiation.holo.coupling_tau =
          numeric_as_double(holo["coupling_tau"], "Radiation.holo.coupling_tau");
    }
    if (has_key(holo, "guard_cells")) {
      radiation.holo.guard_cells =
          strict_int32(holo["guard_cells"], "Radiation.holo.guard_cells");
    }
    if (has_key(holo, "blend_cells")) {
      radiation.holo.blend_cells =
          strict_int32(holo["blend_cells"], "Radiation.holo.blend_cells");
    }
    if (has_key(holo, "min_lo_cells")) {
      radiation.holo.min_lo_cells =
          strict_int32(holo["min_lo_cells"], "Radiation.holo.min_lo_cells");
    }
    if (has_key(holo, "q_min")) {
      radiation.holo.q_min =
          numeric_as_double(holo["q_min"], "Radiation.holo.q_min");
    }
    if (has_key(holo, "q_max")) {
      radiation.holo.q_max =
          numeric_as_double(holo["q_max"], "Radiation.holo.q_max");
    }
    if (holo_has_tau_on) {
      radiation.holo.tau_on =
          numeric_as_double(holo["tau_on"], "Radiation.holo.tau_on");
    }
    if (holo_has_tau_off) {
      radiation.holo.tau_off =
          numeric_as_double(holo["tau_off"], "Radiation.holo.tau_off");
    }
    if (!holo_has_tau_on) {
      radiation.holo.tau_on = radiation.holo.coupling_tau;
    }
    if (has_key(holo, "reduced_flux_on")) {
      radiation.holo.reduced_flux_on = numeric_as_double(
          holo["reduced_flux_on"], "Radiation.holo.reduced_flux_on");
    }
    if (has_key(holo, "reduced_flux_off")) {
      radiation.holo.reduced_flux_off = numeric_as_double(
          holo["reduced_flux_off"], "Radiation.holo.reduced_flux_off");
    }
    if (has_key(holo, "update_interval")) {
      radiation.holo.update_interval = strict_int32(
          holo["update_interval"], "Radiation.holo.update_interval");
    }
    if (has_key(holo, "hold_on")) {
      radiation.holo.hold_on = strict_int32(
          holo["hold_on"], "Radiation.holo.hold_on");
    }
    if (has_key(holo, "min_dwell_steps")) {
      radiation.holo.min_dwell_steps = strict_int32(
          holo["min_dwell_steps"], "Radiation.holo.min_dwell_steps");
    }
    if (has_key(holo, "min_island_cells")) {
      radiation.holo.min_island_cells = strict_int32(
          holo["min_island_cells"], "Radiation.holo.min_island_cells");
    }
    if (has_key(holo, "core_margin_cells")) {
      radiation.holo.core_margin_cells = strict_int32(
          holo["core_margin_cells"], "Radiation.holo.core_margin_cells");
    }
    if (has_key(holo, "solver")) {
      radiation.holo.solver = strict_string(holo["solver"], "Radiation.holo.solver");
    }
    if (has_key(holo, "closure")) {
      radiation.holo.closure = strict_string(holo["closure"], "Radiation.holo.closure");
    }
    if (has_key(holo, "closure_relax")) {
      radiation.holo.closure_relax =
          numeric_as_double(holo["closure_relax"], "Radiation.holo.closure_relax");
    }
    if (has_key(holo, "closure_smooth_passes")) {
      radiation.holo.closure_smooth_passes = strict_int32(
          holo["closure_smooth_passes"], "Radiation.holo.closure_smooth_passes");
    }
    if (has_key(holo, "closure_smooth_alpha")) {
      radiation.holo.closure_smooth_alpha = numeric_as_double(
          holo["closure_smooth_alpha"], "Radiation.holo.closure_smooth_alpha");
    }
    if (has_key(holo, "gamma_alpha")) {
      radiation.holo.consistency_alpha =
          numeric_as_double(holo["gamma_alpha"], "Radiation.holo.gamma_alpha");
    }
    if (has_key(holo, "consistency_alpha")) {
      radiation.holo.consistency_alpha = numeric_as_double(
          holo["consistency_alpha"], "Radiation.holo.consistency_alpha");
    }
    if (has_key(holo, "boundary_flux")) {
      radiation.holo.boundary_flux =
          strict_string(holo["boundary_flux"], "Radiation.holo.boundary_flux");
    }
    if (has_key(holo, "p_rr_tally")) {
      radiation.holo.p_rr_tally =
          strict_bool(holo["p_rr_tally"], "Radiation.holo.p_rr_tally");
    }
    if (has_key(holo, "sn_closure")) {
      radiation.holo.sn_closure =
          strict_bool(holo["sn_closure"], "Radiation.holo.sn_closure");
    }
    if (has_key(holo, "sn_n_angles")) {
      radiation.holo.sn_n_angles =
          strict_int32(holo["sn_n_angles"], "Radiation.holo.sn_n_angles");
    }
    if (has_key(holo, "sn_material_coupling")) {
      radiation.holo.sn_material_coupling = strict_bool(
          holo["sn_material_coupling"], "Radiation.holo.sn_material_coupling");
    }
    if (has_key(holo, "residual_particles_per_cell_group")) {
      radiation.holo.residual_particles_per_cell_group = strict_int32(
          holo["residual_particles_per_cell_group"],
          "Radiation.holo.residual_particles_per_cell_group");
    }
  }

  if (!has_key(kwargs, "boundary")) {
    return;
  }

  const py::handle boundary_obj = kwargs["boundary"];
  if (py::isinstance<py::str>(boundary_obj)) {
    radiation.boundary.type =
        normalize_boundary_mode(strict_string(boundary_obj, "Radiation.boundary"),
                                "Radiation.boundary");
    if (!is_boundary_type(radiation.boundary.type)) {
      throw ValueError(
          "Radiation.boundary must be one of {\"vacuum\", \"reflect\", \"marshak\"}, got " +
          radiation.boundary.type);
    }
    apply_boundary_type_to_faces(radiation.boundary.type);
    return;
  }
  if (!py::isinstance<py::dict>(boundary_obj)) {
    throw_value_type_error("Radiation.boundary", "str|dict", boundary_obj);
  }

  const py::dict boundary = py::reinterpret_borrow<py::dict>(boundary_obj);
  enforce_known_keys(boundary, "Radiation.boundary",
                     {"type", "inner_r", "outer_r", "bottom_z", "top_z",
                      "r_inner", "r_outer", "z_bottom", "z_top",
                      "marshak_particles", "marshak_Tr_eV",
                      "marshak_Tr", "marshak_Tr_map", "marshak_T"});
  if (has_key(boundary, "type")) {
    radiation.boundary.type = normalize_boundary_mode(
        strict_string(boundary["type"], "Radiation.boundary.type"),
        "Radiation.boundary.type");
    if (!is_boundary_type(radiation.boundary.type)) {
      throw ValueError(
          "Radiation.boundary.type must be one of {\"vacuum\", \"reflect\", \"marshak\"}, got " +
          radiation.boundary.type);
    }
    apply_boundary_type_to_faces(radiation.boundary.type);
  }
  const auto parse_face_boundary = [&](const char* canonical_key,
                                       const char* alias_key,
                                       std::string& out,
                                       const std::string& label) {
    const bool has_canonical = has_key(boundary, canonical_key);
    const bool has_alias = has_key(boundary, alias_key);
    if (has_canonical) {
      out = normalize_boundary_mode(
          strict_string(boundary[canonical_key],
                        std::string("Radiation.boundary.") + canonical_key),
          std::string("Radiation.boundary.") + canonical_key);
      if (has_alias) {
        tenryu::core::log_warning(
            std::string("Radiation.boundary.") + alias_key +
            " is ignored because Radiation.boundary." + canonical_key +
            " is also provided");
      }
    } else if (has_alias) {
      out = normalize_boundary_mode(
          strict_string(boundary[alias_key],
                        std::string("Radiation.boundary.") + alias_key),
          std::string("Radiation.boundary.") + alias_key);
    } else {
      return;
    }
    if (!is_boundary_type(out)) {
      throw ValueError("Radiation.boundary." + label +
                       " must be one of {\"vacuum\", \"reflect\", \"marshak\"}, got " + out);
    }
  };
  parse_face_boundary("inner_r", "r_inner", radiation.boundary.inner_r, "inner_r");
  parse_face_boundary("outer_r", "r_outer", radiation.boundary.outer_r, "outer_r");
  parse_face_boundary("bottom_z", "z_bottom", radiation.boundary.bottom_z, "bottom_z");
  parse_face_boundary("top_z", "z_top", radiation.boundary.top_z, "top_z");
  if (has_key(boundary, "marshak_particles")) {
    radiation.boundary.marshak_particles =
        strict_int32(boundary["marshak_particles"],
                     "Radiation.boundary.marshak_particles");
  }
  if (has_key(boundary, "marshak_Tr_eV")) {
    radiation.boundary.marshak_Tr_eV =
        numeric_as_double(boundary["marshak_Tr_eV"], "Radiation.boundary.marshak_Tr_eV");
  }

  if (has_key(boundary, "marshak_T")) {
    tenryu::core::log_warning(
        "Radiation.boundary.marshak_T is deprecated; use marshak_Tr");
    const auto callable =
        extract_callable_or_throw(boundary["marshak_T"], "Radiation.boundary.marshak_T");
    radiation.boundary.marshak_Tr = to_config_callable(callable);
    register_callable("Radiation.boundary.marshak_Tr", callable,
                      boundary["marshak_T"]);
  }
  if (has_key(boundary, "marshak_Tr")) {
    const auto callable = extract_callable_or_throw(boundary["marshak_Tr"],
                                                    "Radiation.boundary.marshak_Tr");
    radiation.boundary.marshak_Tr = to_config_callable(callable);
    register_callable("Radiation.boundary.marshak_Tr", callable,
                      boundary["marshak_Tr"]);
  }
  if (has_key(boundary, "marshak_Tr_map")) {
    const py::handle map_obj = boundary["marshak_Tr_map"];
    if (!py::isinstance<py::dict>(map_obj)) {
      throw_value_type_error("Radiation.boundary.marshak_Tr_map", "dict", map_obj);
    }
    radiation.boundary.marshak_Tr_map.clear();
    for (const auto item : py::reinterpret_borrow<py::dict>(map_obj)) {
      const std::string face = py::str(item.first).cast<std::string>();
      const auto callable = extract_callable_or_throw(
          item.second, "Radiation.boundary.marshak_Tr_map." + face);
      radiation.boundary.marshak_Tr_map[face] = to_config_callable(callable);
      register_callable("Radiation.boundary.marshak_Tr_map." + face, callable,
                        item.second);
    }
  }
}

void Builder::set_laser(py::dict kwargs) {
  mark_block_called(Block::Laser);
  enforce_known_keys(kwargs, "Laser",
                     {"enabled", "wavelength_nm", "mode", "rays_per_beam", "ray_output_count",
                      "ray_output_trajectory", "ray_output_max_steps",
                      "allow_single_material_approximation",
                      "absorption",
                      "lasermesh", "ib", "ra", "raytrace", "raytrace_skip",
                      "raytrace_skip_config",
                      "deposit", "profile", "port_configuration", "cbet", "hot_electron",
                      "beams"});
  if (has_key(kwargs, "allow_single_material_approximation")) {
    warn_ignored_key("Laser.allow_single_material_approximation");
  }

  auto& laser = config.laser;
  if (has_key(kwargs, "enabled")) {
    laser.enabled = strict_bool(kwargs["enabled"], "Laser.enabled");
  }
  if (has_key(kwargs, "wavelength_nm")) {
    laser.wavelength_nm =
        numeric_as_double(kwargs["wavelength_nm"], "Laser.wavelength_nm");
    ensure_positive(laser.wavelength_nm, "Laser.wavelength_nm");
  }
  if (has_key(kwargs, "mode")) {
    laser.mode = strict_string(kwargs["mode"], "Laser.mode");
  }
  if (has_key(kwargs, "rays_per_beam")) {
    laser.rays_per_beam =
        strict_int32(kwargs["rays_per_beam"], "Laser.rays_per_beam");
    laser_rays_per_beam_explicit = true;
  }
  if (has_key(kwargs, "ray_output_count")) {
    laser.ray_output_count =
        strict_int32(kwargs["ray_output_count"], "Laser.ray_output_count");
  }
  if (has_key(kwargs, "ray_output_trajectory")) {
    laser.ray_output_trajectory =
        strict_bool(kwargs["ray_output_trajectory"], "Laser.ray_output_trajectory");
  }
  if (has_key(kwargs, "ray_output_max_steps")) {
    laser.ray_output_max_steps =
        strict_int32(kwargs["ray_output_max_steps"], "Laser.ray_output_max_steps");
  }
  if (has_key(kwargs, "absorption")) {
    const py::handle absorption_obj = kwargs["absorption"];
    if (!py::isinstance<py::dict>(absorption_obj)) {
      throw_value_type_error("Laser.absorption", "dict", absorption_obj);
    }
    const py::dict absorption = py::reinterpret_borrow<py::dict>(absorption_obj);
    enforce_known_keys(absorption, "Laser.absorption",
                       {"model", "eps_n", "eps_crit", "terminate", "coulomb_log_floor",
                        "critical_handling", "debug_dump_lasermesh"});
    if (has_key(absorption, "model")) {
      laser.absorption.model = strict_string(absorption["model"], "Laser.absorption.model");
    }
    if (has_key(absorption, "eps_n")) {
      laser.absorption.eps_n = numeric_as_double(absorption["eps_n"], "Laser.absorption.eps_n");
    }
    if (has_key(absorption, "eps_crit")) {
      laser.raytrace.eps_crit =
          numeric_as_double(absorption["eps_crit"], "Laser.absorption.eps_crit");
    }
    if (has_key(absorption, "terminate")) {
      laser.absorption.terminate =
          strict_bool(absorption["terminate"], "Laser.absorption.terminate");
    }
    if (has_key(absorption, "coulomb_log_floor")) {
      laser.absorption.coulomb_log_floor = numeric_as_double(
          absorption["coulomb_log_floor"], "Laser.absorption.coulomb_log_floor");
    }
    if (has_key(absorption, "debug_dump_lasermesh")) {
      laser.absorption.debug_dump_lasermesh = strict_bool(
          absorption["debug_dump_lasermesh"], "Laser.absorption.debug_dump_lasermesh");
    }
    if (has_key(absorption, "critical_handling")) {
      const py::handle critical_obj = absorption["critical_handling"];
      if (!py::isinstance<py::dict>(critical_obj)) {
        throw_value_type_error("Laser.absorption.critical_handling", "dict", critical_obj);
      }
      const py::dict critical = py::reinterpret_borrow<py::dict>(critical_obj);
      enforce_known_keys(critical, "Laser.absorption.critical_handling",
                         {"eps_n", "eps_crit", "terminate", "terminate_mode"});
      if (has_key(critical, "eps_n")) {
        laser.absorption.eps_n = numeric_as_double(
            critical["eps_n"], "Laser.absorption.critical_handling.eps_n");
      }
      if (has_key(critical, "eps_crit")) {
        laser.raytrace.eps_crit = numeric_as_double(
            critical["eps_crit"], "Laser.absorption.critical_handling.eps_crit");
      }
      if (has_key(critical, "terminate")) {
        laser.absorption.terminate = strict_bool(
            critical["terminate"], "Laser.absorption.critical_handling.terminate");
      }
      if (has_key(critical, "terminate_mode")) {
        laser.absorption.terminate_mode = strict_string(
            critical["terminate_mode"],
            "Laser.absorption.critical_handling.terminate_mode");
        if (laser.absorption.terminate_mode != "escape" &&
            laser.absorption.terminate_mode != "deposit") {
          throw ValueError(
              "Laser.absorption.critical_handling.terminate_mode must be one of "
              "{\"escape\", \"deposit\"}, got " +
              laser.absorption.terminate_mode);
        }
      }
    }
  }
  if (has_key(kwargs, "lasermesh")) {
    const py::handle mesh_obj = kwargs["lasermesh"];
    if (!py::isinstance<py::dict>(mesh_obj)) {
      throw_value_type_error("Laser.lasermesh", "dict", mesh_obj);
    }
    const py::dict lmesh = py::reinterpret_borrow<py::dict>(mesh_obj);
    enforce_known_keys(lmesh, "Laser.lasermesh",
                       {"nr", "nz", "r_max_factor", "z_span_factor", "critical_clip",
                        "critical_margin", "stretch_method", "min_ratio", "mesh_factor",
                        "rmax_n_hat_threshold", "nr_max", "ghost_corona"});
    if (has_key(lmesh, "nr")) {
      laser.lasermesh.nr = strict_int32(lmesh["nr"], "Laser.lasermesh.nr");
    }
    if (has_key(lmesh, "nz")) {
      laser.lasermesh.nz = strict_int32(lmesh["nz"], "Laser.lasermesh.nz");
    }
    if (has_key(lmesh, "r_max_factor")) {
      laser.lasermesh.r_max_factor =
          numeric_as_double(lmesh["r_max_factor"], "Laser.lasermesh.r_max_factor");
    }
    if (has_key(lmesh, "z_span_factor")) {
      laser.lasermesh.z_span_factor =
          numeric_as_double(lmesh["z_span_factor"], "Laser.lasermesh.z_span_factor");
    }
    if (has_key(lmesh, "critical_clip")) {
      laser.lasermesh.critical_clip =
          strict_bool(lmesh["critical_clip"], "Laser.lasermesh.critical_clip");
    }
    if (has_key(lmesh, "critical_margin")) {
      laser.lasermesh.critical_margin =
          numeric_as_double(lmesh["critical_margin"], "Laser.lasermesh.critical_margin");
    }
    if (has_key(lmesh, "stretch_method")) {
      laser.lasermesh.stretch_method =
          strict_string(lmesh["stretch_method"], "Laser.lasermesh.stretch_method");
    }
    if (has_key(lmesh, "min_ratio")) {
      laser.lasermesh.min_ratio =
          numeric_as_double(lmesh["min_ratio"], "Laser.lasermesh.min_ratio");
    }
    if (has_key(lmesh, "mesh_factor")) {
      laser.lasermesh.mesh_factor =
          numeric_as_double(lmesh["mesh_factor"], "Laser.lasermesh.mesh_factor");
    }
    if (has_key(lmesh, "rmax_n_hat_threshold")) {
      laser.lasermesh.rmax_n_hat_threshold = numeric_as_double(
          lmesh["rmax_n_hat_threshold"], "Laser.lasermesh.rmax_n_hat_threshold");
    }
    if (has_key(lmesh, "nr_max")) {
      laser.lasermesh.nr_max = strict_int32(lmesh["nr_max"], "Laser.lasermesh.nr_max");
    }
    if (has_key(lmesh, "ghost_corona")) {
      const py::handle ghost_obj = lmesh["ghost_corona"];
      if (!py::isinstance<py::dict>(ghost_obj)) {
        throw_value_type_error("Laser.lasermesh.ghost_corona", "dict", ghost_obj);
      }
      const py::dict ghost = py::reinterpret_borrow<py::dict>(ghost_obj);
      enforce_known_keys(ghost, "Laser.lasermesh.ghost_corona",
                         {"enabled", "n_out", "ne_min_frac", "ne_max_frac", "Te_min_eV",
                          "zbar_min", "zbar_max", "handoff_cells", "handoff_decay",
                          "transition_enabled", "transition_resolved_nhat",
                          "transition_resolved_cells", "transition_density_exponent"});
      if (has_key(ghost, "enabled")) {
        laser.lasermesh.ghost_corona.enabled =
            strict_bool(ghost["enabled"], "Laser.lasermesh.ghost_corona.enabled");
      }
      if (has_key(ghost, "n_out")) {
        laser.lasermesh.ghost_corona.n_out =
            strict_int32(ghost["n_out"], "Laser.lasermesh.ghost_corona.n_out");
      }
      if (has_key(ghost, "ne_min_frac")) {
        laser.lasermesh.ghost_corona.ne_min_frac = numeric_as_double(
            ghost["ne_min_frac"], "Laser.lasermesh.ghost_corona.ne_min_frac");
      }
      if (has_key(ghost, "ne_max_frac")) {
        laser.lasermesh.ghost_corona.ne_max_frac = numeric_as_double(
            ghost["ne_max_frac"], "Laser.lasermesh.ghost_corona.ne_max_frac");
      }
      if (has_key(ghost, "Te_min_eV")) {
        laser.lasermesh.ghost_corona.Te_min_eV = numeric_as_double(
            ghost["Te_min_eV"], "Laser.lasermesh.ghost_corona.Te_min_eV");
      }
      if (has_key(ghost, "zbar_min")) {
        laser.lasermesh.ghost_corona.zbar_min = numeric_as_double(
            ghost["zbar_min"], "Laser.lasermesh.ghost_corona.zbar_min");
      }
      if (has_key(ghost, "zbar_max")) {
        laser.lasermesh.ghost_corona.zbar_max = numeric_as_double(
            ghost["zbar_max"], "Laser.lasermesh.ghost_corona.zbar_max");
      }
      if (has_key(ghost, "handoff_cells")) {
        laser.lasermesh.ghost_corona.handoff_cells = strict_int32(
            ghost["handoff_cells"], "Laser.lasermesh.ghost_corona.handoff_cells");
      }
      if (has_key(ghost, "handoff_decay")) {
        laser.lasermesh.ghost_corona.handoff_decay = numeric_as_double(
            ghost["handoff_decay"], "Laser.lasermesh.ghost_corona.handoff_decay");
      }
      if (has_key(ghost, "transition_enabled")) {
        laser.lasermesh.ghost_corona.transition_enabled = strict_bool(
            ghost["transition_enabled"],
            "Laser.lasermesh.ghost_corona.transition_enabled");
      }
      if (has_key(ghost, "transition_resolved_nhat")) {
        laser.lasermesh.ghost_corona.transition_resolved_nhat = numeric_as_double(
            ghost["transition_resolved_nhat"],
            "Laser.lasermesh.ghost_corona.transition_resolved_nhat");
      }
      if (has_key(ghost, "transition_resolved_cells")) {
        laser.lasermesh.ghost_corona.transition_resolved_cells = strict_int32(
            ghost["transition_resolved_cells"],
            "Laser.lasermesh.ghost_corona.transition_resolved_cells");
      }
      if (has_key(ghost, "transition_density_exponent")) {
        laser.lasermesh.ghost_corona.transition_density_exponent = numeric_as_double(
            ghost["transition_density_exponent"],
            "Laser.lasermesh.ghost_corona.transition_density_exponent");
      }
    }
  }
  if (has_key(kwargs, "ib")) {
    const py::handle ib_obj = kwargs["ib"];
    if (!py::isinstance<py::dict>(ib_obj)) {
      throw_value_type_error("Laser.ib", "dict", ib_obj);
    }
    const py::dict ib = py::reinterpret_borrow<py::dict>(ib_obj);
    enforce_known_keys(
        ib, "Laser.ib",
        {"zeff_model", "species", "coulomb_log_model", "langdon_model",
         "langdon_te_min_eV"});
    if (has_key(ib, "zeff_model")) {
      laser.ib.zeff_model =
          strict_string(ib["zeff_model"], "Laser.ib.zeff_model");
      if (laser.ib.zeff_model != "auto" &&
          laser.ib.zeff_model != "off" &&
          laser.ib.zeff_model != "sequential_strip" &&
          laser.ib.zeff_model != "table") {
        throw ValueError(
            "Laser.ib.zeff_model must be one of "
            "{\"auto\", \"off\", \"sequential_strip\", \"table\"}");
      }
    }
    if (has_key(ib, "species")) {
      const py::handle species_obj = ib["species"];
      if (!py::isinstance<py::sequence>(species_obj) ||
          py::isinstance<py::str>(species_obj)) {
        throw_value_type_error("Laser.ib.species", "list[list[float]]",
                               species_obj);
      }
      const py::sequence species =
          py::reinterpret_borrow<py::sequence>(species_obj);
      if (species.size() > 4) {
        throw ValueError("Laser.ib.species supports at most 4 entries");
      }
      laser.ib.species_z.clear();
      laser.ib.species_x.clear();
      laser.ib.species_z.reserve(species.size());
      laser.ib.species_x.reserve(species.size());
      double x_sum = 0.0;
      for (std::size_t i = 0; i < species.size(); ++i) {
        const std::string path =
            "Laser.ib.species[" + std::to_string(i) + "]";
        const py::handle entry_obj = species[i];
        if (!py::isinstance<py::sequence>(entry_obj) ||
            py::isinstance<py::str>(entry_obj)) {
          throw_value_type_error(path, "list[float]", entry_obj);
        }
        const py::sequence entry =
            py::reinterpret_borrow<py::sequence>(entry_obj);
        if (entry.size() != 2) {
          throw ValueError(path + " must have exactly 2 elements");
        }
        const double z_nuc =
            numeric_as_double(entry[0], path + "[0]");
        const double x_frac =
            numeric_as_double(entry[1], path + "[1]");
        if (!laser.ib.species_z.empty() &&
            !(z_nuc > laser.ib.species_z.back())) {
          throw ValueError(
              "Laser.ib.species nuclear charges must be strictly ascending");
        }
        if (!(x_frac > 0.0)) {
          throw ValueError(
              "Laser.ib.species number fractions must be > 0");
        }
        laser.ib.species_z.push_back(z_nuc);
        laser.ib.species_x.push_back(x_frac);
        x_sum += x_frac;
      }
      if (!laser.ib.species_x.empty() &&
          !(x_sum >= 0.99 && x_sum <= 1.01)) {
        throw ValueError(
            "Laser.ib.species number fractions must sum within [0.99, 1.01]");
      }
    }
    if (has_key(ib, "coulomb_log_model")) {
      laser.ib.coulomb_log_model = strict_string(
          ib["coulomb_log_model"], "Laser.ib.coulomb_log_model");
      if (laser.ib.coulomb_log_model != "debye" &&
          laser.ib.coulomb_log_model != "laser_frequency") {
        throw ValueError(
            "Laser.ib.coulomb_log_model must be \"debye\" or "
            "\"laser_frequency\"");
      }
    }
    if (has_key(ib, "langdon_model")) {
      laser.ib.langdon_model =
          strict_string(ib["langdon_model"], "Laser.ib.langdon_model");
      if (laser.ib.langdon_model != "auto" &&
          laser.ib.langdon_model != "off" &&
          laser.ib.langdon_model != "legacy_vacuum_map") {
        throw ValueError(
            "Laser.ib.langdon_model must be one of "
            "{\"auto\", \"off\", \"legacy_vacuum_map\"}");
      }
    }
    if (has_key(ib, "langdon_te_min_eV")) {
      laser.ib.langdon_te_min_eV = numeric_as_double(
          ib["langdon_te_min_eV"], "Laser.ib.langdon_te_min_eV");
      if (!(laser.ib.langdon_te_min_eV >= 0.0 &&
            laser.ib.langdon_te_min_eV <= 1.0e5)) {
        throw ValueError("Laser.ib.langdon_te_min_eV must be in [0, 1e5]");
      }
    }
    if (laser.ib.zeff_model == "sequential_strip" &&
        laser.ib.species_z.empty()) {
      throw ConfigError(
          "Laser.ib.species is required when zeff_model=sequential_strip");
    }
  }
  if (has_key(kwargs, "ra")) {
    const py::handle ra_obj = kwargs["ra"];
    if (!py::isinstance<py::dict>(ra_obj)) {
      throw_value_type_error("Laser.ra", "dict", ra_obj);
    }
    const py::dict ra = py::reinterpret_borrow<py::dict>(ra_obj);
    enforce_known_keys(ra, "Laser.ra", {"enable", "chi_p", "c_ra"});
    if (has_key(ra, "enable")) {
      laser.ra.enable = strict_bool(ra["enable"], "Laser.ra.enable");
    }
    if (has_key(ra, "chi_p")) {
      laser.ra.chi_p =
          numeric_as_double(ra["chi_p"], "Laser.ra.chi_p");
      if (!(laser.ra.chi_p >= 0.0 && laser.ra.chi_p <= 1.0)) {
        throw ValueError("Laser.ra.chi_p must be in [0, 1]");
      }
    }
    if (has_key(ra, "c_ra")) {
      laser.ra.c_ra =
          numeric_as_double(ra["c_ra"], "Laser.ra.c_ra");
      if (!(laser.ra.c_ra >= 0.0 && laser.ra.c_ra <= 10.0)) {
        throw ValueError("Laser.ra.c_ra must be in [0, 10]");
      }
    }
  }
  if (has_key(kwargs, "raytrace")) {
    const py::handle trace_obj = kwargs["raytrace"];
    if (!py::isinstance<py::dict>(trace_obj)) {
      throw_value_type_error("Laser.raytrace", "dict", trace_obj);
    }
    const py::dict raytrace = py::reinterpret_borrow<py::dict>(trace_obj);
    enforce_known_keys(raytrace, "Laser.raytrace",
                       {"cfl_ray", "intensity_cutoff", "eps_crit", "max_steps", "integrator",
                        "test_kappa", "ds_adapt_g_target", "ds_adapt_tau_target",
                        "ds_adapt_theta_target", "ds_adapt_max_factor", "debug_one_ray"});
    if (has_key(raytrace, "cfl_ray")) {
      laser.raytrace.cfl_ray =
          numeric_as_double(raytrace["cfl_ray"], "Laser.raytrace.cfl_ray");
    }
    if (has_key(raytrace, "intensity_cutoff")) {
      laser.raytrace.intensity_cutoff = numeric_as_double(raytrace["intensity_cutoff"],
                                                          "Laser.raytrace.intensity_cutoff");
    }
    if (has_key(raytrace, "eps_crit")) {
      laser.raytrace.eps_crit =
          numeric_as_double(raytrace["eps_crit"], "Laser.raytrace.eps_crit");
    }
    if (has_key(raytrace, "max_steps")) {
      laser.raytrace.max_steps =
          strict_int32(raytrace["max_steps"], "Laser.raytrace.max_steps");
    }
    if (has_key(raytrace, "integrator")) {
      laser.raytrace.integrator =
          strict_string(raytrace["integrator"], "Laser.raytrace.integrator");
    }
    if (has_key(raytrace, "test_kappa")) {
      laser.raytrace.test_kappa =
          numeric_as_double(raytrace["test_kappa"], "Laser.raytrace.test_kappa");
    }
    if (has_key(raytrace, "ds_adapt_g_target")) {
      laser.raytrace.ds_adapt_g_target = numeric_as_double(
          raytrace["ds_adapt_g_target"], "Laser.raytrace.ds_adapt_g_target");
    }
    if (has_key(raytrace, "ds_adapt_tau_target")) {
      laser.raytrace.ds_adapt_tau_target = numeric_as_double(
          raytrace["ds_adapt_tau_target"], "Laser.raytrace.ds_adapt_tau_target");
    }
    if (has_key(raytrace, "ds_adapt_theta_target")) {
      laser.raytrace.ds_adapt_theta_target = numeric_as_double(
          raytrace["ds_adapt_theta_target"], "Laser.raytrace.ds_adapt_theta_target");
    }
    if (has_key(raytrace, "ds_adapt_max_factor")) {
      laser.raytrace.ds_adapt_max_factor = numeric_as_double(
          raytrace["ds_adapt_max_factor"], "Laser.raytrace.ds_adapt_max_factor");
    }
    if (has_key(raytrace, "debug_one_ray")) {
      laser.raytrace.debug_one_ray =
          strict_bool(raytrace["debug_one_ray"], "Laser.raytrace.debug_one_ray");
    }
  }
  auto parse_skip_config = [&](const py::dict& skip_cfg, const std::string& path) {
    enforce_known_keys(skip_cfg, path,
                       {"enabled", "threshold", "max_consecutive", "norm", "crit_guard"});
    if (has_key(skip_cfg, "enabled")) {
      laser.raytrace_skip_config.enabled =
          strict_bool(skip_cfg["enabled"], path + ".enabled");
    }
    if (has_key(skip_cfg, "threshold")) {
      laser.raytrace_skip_config.threshold =
          numeric_as_double(skip_cfg["threshold"], path + ".threshold");
    }
    if (has_key(skip_cfg, "max_consecutive")) {
      laser.raytrace_skip_config.max_consecutive =
          strict_int32(skip_cfg["max_consecutive"], path + ".max_consecutive");
    }
    if (has_key(skip_cfg, "norm")) {
      laser.raytrace_skip_config.norm =
          strict_string(skip_cfg["norm"], path + ".norm");
    }
    if (has_key(skip_cfg, "crit_guard")) {
      laser.raytrace_skip_config.crit_guard =
          numeric_as_double(skip_cfg["crit_guard"], path + ".crit_guard");
    }
    laser.raytrace_skip = laser.raytrace_skip_config.enabled
                              ? laser.raytrace_skip_config.threshold
                              : 0.0;
  };

  if (has_key(kwargs, "raytrace_skip")) {
    const py::handle skip_obj = kwargs["raytrace_skip"];
    if (py::isinstance<py::dict>(skip_obj)) {
      parse_skip_config(py::reinterpret_borrow<py::dict>(skip_obj), "Laser.raytrace_skip");
    } else {
      laser.raytrace_skip = numeric_as_double(skip_obj, "Laser.raytrace_skip");
      if (laser.raytrace_skip > 0.0) {
        laser.raytrace_skip_config.enabled = true;
        laser.raytrace_skip_config.threshold = laser.raytrace_skip;
      } else {
        laser.raytrace_skip_config.enabled = false;
      }
    }
  }
  if (has_key(kwargs, "raytrace_skip_config")) {
    const py::handle skip_cfg_obj = kwargs["raytrace_skip_config"];
    if (!py::isinstance<py::dict>(skip_cfg_obj)) {
      throw_value_type_error("Laser.raytrace_skip_config", "dict", skip_cfg_obj);
    }
    parse_skip_config(py::reinterpret_borrow<py::dict>(skip_cfg_obj),
                      "Laser.raytrace_skip_config");
  }
  if (has_key(kwargs, "deposit")) {
    const py::handle deposit_obj = kwargs["deposit"];
    if (!py::isinstance<py::dict>(deposit_obj)) {
      throw_value_type_error("Laser.deposit", "dict", deposit_obj);
    }
    const py::dict deposit = py::reinterpret_borrow<py::dict>(deposit_obj);
    enforce_known_keys(deposit, "Laser.deposit",
                       {"conservation_tol", "deposit_smooth_passes",
                        "deposit_smooth_alpha"});
    if (has_key(deposit, "conservation_tol")) {
      laser.deposit.conservation_tol =
          numeric_as_double(deposit["conservation_tol"], "Laser.deposit.conservation_tol");
    }
    if (has_key(deposit, "deposit_smooth_passes")) {
      laser.deposit.deposit_smooth_passes = strict_int32(
          deposit["deposit_smooth_passes"], "Laser.deposit.deposit_smooth_passes");
    }
    if (has_key(deposit, "deposit_smooth_alpha")) {
      laser.deposit.deposit_smooth_alpha = numeric_as_double(
          deposit["deposit_smooth_alpha"], "Laser.deposit.deposit_smooth_alpha");
    }
  }

  if (has_key(kwargs, "port_configuration")) {
    const py::handle port_config_obj = kwargs["port_configuration"];
    if (!py::isinstance<py::dict>(port_config_obj)) {
      throw_value_type_error("Laser.port_configuration", "dict", port_config_obj);
    }
    const py::dict port_config =
        py::reinterpret_borrow<py::dict>(port_config_obj);
    enforce_known_keys(port_config, "Laser.port_configuration",
                       {"normalization", "ports"});
    if (has_key(port_config, "normalization")) {
      laser.port_configuration.normalization = strict_string(
          port_config["normalization"], "Laser.port_configuration.normalization");
    }
    if (has_key(port_config, "ports")) {
      const py::handle ports_obj = port_config["ports"];
      if (!py::isinstance<py::sequence>(ports_obj) ||
          py::isinstance<py::str>(ports_obj)) {
        throw_value_type_error("Laser.port_configuration.ports",
                               "list[dict]", ports_obj);
      }
      const py::sequence ports =
          py::reinterpret_borrow<py::sequence>(ports_obj);
      laser.port_configuration.ports.clear();
      laser.port_configuration.ports.reserve(ports.size());
      for (std::size_t i = 0; i < ports.size(); ++i) {
        const py::handle port_obj = ports[i];
        const std::string path =
            "Laser.port_configuration.ports[" + std::to_string(i) + "]";
        if (!py::isinstance<py::dict>(port_obj)) {
          throw_value_type_error(path, "dict", port_obj);
        }
        const py::dict port = py::reinterpret_borrow<py::dict>(port_obj);
        enforce_known_keys(
            port, path,
            {"port_id", "direction", "roll_deg", "power_weight",
             "delta_lambda_nm", "beam_class", "polarization"});

        Config::LaserConfig::LaserPortConfig out;
        if (has_key(port, "port_id")) {
          out.port_id = strict_int32(port["port_id"], path + ".port_id");
        }
        if (has_key(port, "direction")) {
          out.direction =
              strict_double_vector(port["direction"], path + ".direction");
          if (out.direction.size() != 3) {
            throw ValueError(path + ".direction must have exactly 3 elements");
          }
        }
        if (has_key(port, "roll_deg")) {
          out.roll_deg =
              numeric_as_double(port["roll_deg"], path + ".roll_deg");
        }
        if (has_key(port, "power_weight")) {
          out.power_weight =
              numeric_as_double(port["power_weight"], path + ".power_weight");
        }
        if (has_key(port, "delta_lambda_nm")) {
          out.delta_lambda_nm = numeric_as_double(
              port["delta_lambda_nm"], path + ".delta_lambda_nm");
        }
        if (has_key(port, "beam_class")) {
          out.beam_class =
              strict_string(port["beam_class"], path + ".beam_class");
        }
        if (has_key(port, "polarization")) {
          out.polarization =
              strict_string(port["polarization"], path + ".polarization");
        }
        if (out.direction.size() == 3) {
          const double norm =
              std::sqrt(out.direction[0] * out.direction[0] +
                        out.direction[1] * out.direction[1] +
                        out.direction[2] * out.direction[2]);
          if (norm > 0.0) {
            for (double& component : out.direction) {
              component /= norm;
            }
          }
        }
        laser.port_configuration.ports.push_back(std::move(out));
      }
    }
  }

  if (has_key(kwargs, "cbet")) {
    const py::handle cbet_obj = kwargs["cbet"];
    if (!py::isinstance<py::dict>(cbet_obj)) {
      throw_value_type_error("Laser.cbet", "dict", cbet_obj);
    }
    const py::dict cbet = py::reinterpret_borrow<py::dict>(cbet_obj);
    enforce_known_keys(cbet, "Laser.cbet",
                       {"enable", "f_cbet", "alpha_iaw", "theta_cap", "tol", "max_iters",
                        "n_impact_bins", "n_phi", "ne_frac_cutoff", "k_a_floor",
                        "max_segments_per_ray", "test_chi", "geometry_mode",
                        "n_section_phi"});
    if (has_key(cbet, "enable")) {
      laser.cbet.enable = strict_bool(cbet["enable"], "Laser.cbet.enable");
    }
    if (has_key(cbet, "f_cbet")) {
      laser.cbet.f_cbet = numeric_as_double(cbet["f_cbet"], "Laser.cbet.f_cbet");
      ensure_positive(laser.cbet.f_cbet, "Laser.cbet.f_cbet");
    }
    if (has_key(cbet, "alpha_iaw")) {
      laser.cbet.alpha_iaw = numeric_as_double(cbet["alpha_iaw"], "Laser.cbet.alpha_iaw");
      ensure_positive(laser.cbet.alpha_iaw, "Laser.cbet.alpha_iaw");
    }
    if (has_key(cbet, "theta_cap")) {
      laser.cbet.theta_cap = numeric_as_double(cbet["theta_cap"], "Laser.cbet.theta_cap");
      if (!(laser.cbet.theta_cap > 0.0) || laser.cbet.theta_cap >= 1.0) {
        throw ValueError("Laser.cbet.theta_cap must be in (0, 1)");
      }
    }
    if (has_key(cbet, "tol")) {
      laser.cbet.tol = numeric_as_double(cbet["tol"], "Laser.cbet.tol");
      ensure_positive(laser.cbet.tol, "Laser.cbet.tol");
    }
    if (has_key(cbet, "max_iters")) {
      laser.cbet.max_iters = strict_int32(cbet["max_iters"], "Laser.cbet.max_iters");
      if (laser.cbet.max_iters < 1) {
        throw ValueError("Laser.cbet.max_iters must be >= 1");
      }
    }
    if (has_key(cbet, "n_impact_bins")) {
      laser.cbet.n_impact_bins =
          strict_int32(cbet["n_impact_bins"], "Laser.cbet.n_impact_bins");
      if (laser.cbet.n_impact_bins < 1) {
        throw ValueError("Laser.cbet.n_impact_bins must be >= 1");
      }
    }
    if (has_key(cbet, "n_phi")) {
      laser.cbet.n_phi = strict_int32(cbet["n_phi"], "Laser.cbet.n_phi");
      if (laser.cbet.n_phi < 1) {
        throw ValueError("Laser.cbet.n_phi must be >= 1");
      }
    }
    if (has_key(cbet, "ne_frac_cutoff")) {
      laser.cbet.ne_frac_cutoff =
          numeric_as_double(cbet["ne_frac_cutoff"], "Laser.cbet.ne_frac_cutoff");
      if (!(laser.cbet.ne_frac_cutoff > 0.0) || laser.cbet.ne_frac_cutoff > 1.0) {
        throw ValueError("Laser.cbet.ne_frac_cutoff must be in (0, 1]");
      }
    }
    if (has_key(cbet, "k_a_floor")) {
      laser.cbet.k_a_floor = numeric_as_double(cbet["k_a_floor"], "Laser.cbet.k_a_floor");
      ensure_positive(laser.cbet.k_a_floor, "Laser.cbet.k_a_floor");
    }
    if (has_key(cbet, "max_segments_per_ray")) {
      laser.cbet.max_segments_per_ray =
          strict_int32(cbet["max_segments_per_ray"], "Laser.cbet.max_segments_per_ray");
      if (laser.cbet.max_segments_per_ray < 0) {
        throw ValueError("Laser.cbet.max_segments_per_ray must be >= 0");
      }
    }
    if (has_key(cbet, "test_chi")) {
      laser.cbet.test_chi = numeric_as_double(cbet["test_chi"], "Laser.cbet.test_chi");
    }
    if (has_key(cbet, "geometry_mode")) {
      laser.cbet.geometry_mode =
          strict_string(cbet["geometry_mode"], "Laser.cbet.geometry_mode");
    }
    if (has_key(cbet, "n_section_phi")) {
      laser.cbet.n_section_phi =
          strict_int32(cbet["n_section_phi"], "Laser.cbet.n_section_phi");
    }
  }

  if (has_key(kwargs, "hot_electron")) {
    const py::handle he_obj = kwargs["hot_electron"];
    if (!py::isinstance<py::dict>(he_obj)) {
      throw_value_type_error("Laser.hot_electron", "dict", he_obj);
    }
    const py::dict he = py::reinterpret_borrow<py::dict>(he_obj);
    enforce_known_keys(he, "Laser.hot_electron",
                       {"enable", "source_nc_fraction", "eta_hot", "eta_hot_table",
                        "eta_mode", "eta_model", "tpd_overlap_mode",
                        "srs_overlap_mode", "illumination_metric",
                        "common_wave_delta_theta_deg",
                        "T_hot_eV", "n_energy_groups", "E_min_over_Th", "E_max_over_Th",
                        "angular_model", "theta_div_deg", "n_mu", "n_phi",
                        "subtract_from_laser", "inner_bc", "explicit_source_limit",
                        "sources"});
    if (has_key(he, "enable")) {
      laser.hot_electron.enable = strict_bool(he["enable"], "Laser.hot_electron.enable");
    }
    if (has_key(he, "source_nc_fraction")) {
      laser.hot_electron.source_nc_fraction =
          numeric_as_double(he["source_nc_fraction"], "Laser.hot_electron.source_nc_fraction");
      if (!(laser.hot_electron.source_nc_fraction > 0.0) ||
          laser.hot_electron.source_nc_fraction > 1.0) {
        throw ValueError("Laser.hot_electron.source_nc_fraction must be in (0, 1]");
      }
    }
    if (has_key(he, "eta_hot")) {
      laser.hot_electron.eta_hot = numeric_as_double(he["eta_hot"], "Laser.hot_electron.eta_hot");
      if (laser.hot_electron.eta_hot < 0.0 || laser.hot_electron.eta_hot > 0.95) {
        throw ValueError("Laser.hot_electron.eta_hot must be in [0, 0.95]");
      }
    }
    if (has_key(he, "eta_hot_table")) {
      const auto callable =
          extract_callable_or_throw(he["eta_hot_table"], "Laser.hot_electron.eta_hot_table");
      laser.hot_electron.eta_hot_table = to_config_callable(callable);
      register_callable("Laser.hot_electron.eta_hot_table", callable, he["eta_hot_table"]);
    }
    if (has_key(he, "eta_mode")) {
      laser.hot_electron.eta_mode =
          strict_string(he["eta_mode"], "Laser.hot_electron.eta_mode");
    }
    if (has_key(he, "tpd_overlap_mode")) {
      laser.hot_electron.tpd_overlap_mode = strict_string(
          he["tpd_overlap_mode"], "Laser.hot_electron.tpd_overlap_mode");
    }
    if (has_key(he, "srs_overlap_mode")) {
      laser.hot_electron.srs_overlap_mode = strict_string(
          he["srs_overlap_mode"], "Laser.hot_electron.srs_overlap_mode");
    }
    if (has_key(he, "illumination_metric")) {
      laser.hot_electron.illumination_metric = strict_string(
          he["illumination_metric"], "Laser.hot_electron.illumination_metric");
    }
    if (has_key(he, "common_wave_delta_theta_deg")) {
      laser.hot_electron.common_wave_delta_theta_deg = numeric_as_double(
          he["common_wave_delta_theta_deg"],
          "Laser.hot_electron.common_wave_delta_theta_deg");
    }
    if (has_key(he, "eta_model")) {
      const py::handle eta_model_obj = he["eta_model"];
      if (!py::isinstance<py::dict>(eta_model_obj)) {
        throw_value_type_error("Laser.hot_electron.eta_model", "dict", eta_model_obj);
      }
      const py::dict eta_model = py::reinterpret_borrow<py::dict>(eta_model_obj);
      enforce_known_keys(eta_model, "Laser.hot_electron.eta_model",
                         {"ln_filter_tau_s", "eta_total_cap"});
      if (has_key(eta_model, "ln_filter_tau_s")) {
        laser.hot_electron.eta_model.ln_filter_tau_s =
            numeric_as_double(eta_model["ln_filter_tau_s"],
                              "Laser.hot_electron.eta_model.ln_filter_tau_s");
      }
      if (has_key(eta_model, "eta_total_cap")) {
        laser.hot_electron.eta_model.eta_total_cap =
            numeric_as_double(eta_model["eta_total_cap"],
                              "Laser.hot_electron.eta_model.eta_total_cap");
      }
    }
    if (has_key(he, "T_hot_eV")) {
      laser.hot_electron.T_hot_eV = numeric_as_double(he["T_hot_eV"], "Laser.hot_electron.T_hot_eV");
      if (!(laser.hot_electron.T_hot_eV > 0.0)) {
        throw ValueError("Laser.hot_electron.T_hot_eV must be > 0");
      }
    }
    if (has_key(he, "n_energy_groups")) {
      laser.hot_electron.n_energy_groups =
          strict_int32(he["n_energy_groups"], "Laser.hot_electron.n_energy_groups");
      if (laser.hot_electron.n_energy_groups < 1) {
        throw ValueError("Laser.hot_electron.n_energy_groups must be >= 1");
      }
    }
    if (has_key(he, "E_min_over_Th")) {
      laser.hot_electron.E_min_over_Th =
          numeric_as_double(he["E_min_over_Th"], "Laser.hot_electron.E_min_over_Th");
      if (!(laser.hot_electron.E_min_over_Th > 0.0)) {
        throw ValueError("Laser.hot_electron.E_min_over_Th must be > 0");
      }
    }
    if (has_key(he, "E_max_over_Th")) {
      laser.hot_electron.E_max_over_Th =
          numeric_as_double(he["E_max_over_Th"], "Laser.hot_electron.E_max_over_Th");
    }
    if (!(laser.hot_electron.E_max_over_Th > laser.hot_electron.E_min_over_Th)) {
      throw ValueError("Laser.hot_electron.E_max_over_Th must be > E_min_over_Th");
    }
    if (has_key(he, "angular_model")) {
      laser.hot_electron.angular_model =
          strict_string(he["angular_model"], "Laser.hot_electron.angular_model");
      if (laser.hot_electron.angular_model != "cone" &&
          laser.hot_electron.angular_model != "radial") {
        throw ValueError("Laser.hot_electron.angular_model must be \"cone\" or \"radial\"");
      }
    }
    if (has_key(he, "theta_div_deg")) {
      laser.hot_electron.theta_div_deg =
          numeric_as_double(he["theta_div_deg"], "Laser.hot_electron.theta_div_deg");
      if (laser.hot_electron.theta_div_deg < 0.0 ||
          laser.hot_electron.theta_div_deg > 90.0) {
        throw ValueError("Laser.hot_electron.theta_div_deg must be in [0, 90]");
      }
    }
    if (has_key(he, "n_mu")) {
      laser.hot_electron.n_mu = strict_int32(he["n_mu"], "Laser.hot_electron.n_mu");
      if (laser.hot_electron.n_mu < 1) {
        throw ValueError("Laser.hot_electron.n_mu must be >= 1");
      }
    }
    if (has_key(he, "n_phi")) {
      laser.hot_electron.n_phi = strict_int32(he["n_phi"], "Laser.hot_electron.n_phi");
      if (laser.hot_electron.n_phi < 1) {
        throw ValueError("Laser.hot_electron.n_phi must be >= 1");
      }
    }
    if (has_key(he, "subtract_from_laser")) {
      laser.hot_electron.subtract_from_laser =
          strict_bool(he["subtract_from_laser"], "Laser.hot_electron.subtract_from_laser");
    }
    if (has_key(he, "inner_bc")) {
      laser.hot_electron.inner_bc = strict_string(he["inner_bc"], "Laser.hot_electron.inner_bc");
      if (laser.hot_electron.inner_bc != "deposit_residual" &&
          laser.hot_electron.inner_bc != "escape") {
        throw ValueError(
            "Laser.hot_electron.inner_bc must be \"deposit_residual\" or \"escape\"");
      }
    }
    if (has_key(he, "explicit_source_limit")) {
      laser.hot_electron.explicit_source_limit =
          numeric_as_double(he["explicit_source_limit"], "Laser.hot_electron.explicit_source_limit");
      if (!(laser.hot_electron.explicit_source_limit > 0.0)) {
        throw ValueError("Laser.hot_electron.explicit_source_limit must be > 0");
      }
    }
    if (has_key(he, "sources")) {
      static const char* kShorthandKeys[] = {
          "source_nc_fraction", "eta_hot", "eta_hot_table", "T_hot_eV",
          "n_energy_groups", "E_min_over_Th", "E_max_over_Th",
          "theta_div_deg", "n_mu", "n_phi"};
      for (const char* shorthand_key : kShorthandKeys) {
        if (has_key(he, shorthand_key)) {
          throw ValueError(std::string("Laser.hot_electron.sources cannot be "
                                       "combined with the scalar shorthand key \"") +
                           shorthand_key + "\"");
        }
      }
      const py::handle sources_obj = he["sources"];
      if (!py::isinstance<py::sequence>(sources_obj) ||
          py::isinstance<py::str>(sources_obj)) {
        throw_value_type_error("Laser.hot_electron.sources", "list of dicts", sources_obj);
      }
      const py::sequence sources_seq = py::reinterpret_borrow<py::sequence>(sources_obj);
      const std::size_t n_sources = sources_seq.size();
      if (n_sources == 0) {
        throw ValueError("Laser.hot_electron.sources must contain at least one channel");
      }
      if (n_sources > static_cast<std::size_t>(
                          Config::LaserConfig::HotElectronConfig::kMaxSources)) {
        throw ValueError("Laser.hot_electron.sources supports at most 4 channels");
      }
      laser.hot_electron.sources_specified = true;
      laser.hot_electron.sources.clear();
      laser.hot_electron.sources.reserve(n_sources);
      for (std::size_t si = 0; si < n_sources; ++si) {
        const std::string path = "Laser.hot_electron.sources[" + std::to_string(si) + "]";
        const py::handle ch_obj = sources_seq[si];
        if (!py::isinstance<py::dict>(ch_obj)) {
          throw_value_type_error(path, "dict", ch_obj);
        }
        const py::dict ch = py::reinterpret_borrow<py::dict>(ch_obj);
        enforce_known_keys(ch, path,
                           {"mechanism", "capture_nc_fraction", "eta", "eta_table",
                            "eval_nc_fraction", "threshold_multiplier", "eta_inf",
                            "eta_hard_cap", "shape_coefficient", "relaxation_model",
                            "relaxation_tau_s", "relaxation_tau_min_s",
                            "relaxation_tau_max_s",
                            "T_hot_eV", "n_energy_groups", "E_min_over_Th",
                            "E_max_over_Th", "theta_div_deg", "tpd_theta_deg",
                            "tpd_delta_deg", "n_mu", "n_phi"});
        Config::LaserConfig::HotEChannelConfig channel;
        if (has_key(ch, "mechanism")) {
          channel.mechanism = strict_string(ch["mechanism"], path + ".mechanism");
          if (channel.mechanism != "cone" && channel.mechanism != "tpd" &&
              channel.mechanism != "srs") {
            throw ValueError(path + ".mechanism must be \"cone\", \"tpd\", or \"srs\"");
          }
        }
        if (channel.mechanism == "tpd" && has_key(ch, "theta_div_deg")) {
          throw ValueError(path + ".theta_div_deg is not valid for mechanism \"tpd\""
                           " (use tpd_theta_deg / tpd_delta_deg)");
        }
        if (channel.mechanism != "tpd" &&
            (has_key(ch, "tpd_theta_deg") || has_key(ch, "tpd_delta_deg"))) {
          throw ValueError(path + ".tpd_theta_deg / tpd_delta_deg are only valid"
                           " for mechanism \"tpd\"");
        }
        if (channel.mechanism == "srs") {
          channel.capture_nc_fraction = 0.18;
          channel.T_hot_eV = 4.5e4;
          channel.theta_div_deg = 20.0;
        } else if (channel.mechanism == "tpd") {
          channel.T_hot_eV = 6.0e4;
        }
        if (has_key(ch, "capture_nc_fraction")) {
          channel.capture_nc_fraction =
              numeric_as_double(ch["capture_nc_fraction"], path + ".capture_nc_fraction");
        }
        if (!(channel.capture_nc_fraction > 0.0) || channel.capture_nc_fraction > 1.0) {
          throw ValueError(path + ".capture_nc_fraction must be in (0, 1]");
        }
        if (has_key(ch, "eta")) {
          channel.eta = numeric_as_double(ch["eta"], path + ".eta");
          if (channel.eta < 0.0 || channel.eta > 0.95) {
            throw ValueError(path + ".eta must be in [0, 0.95]");
          }
        }
        if (has_key(ch, "eta_table")) {
          const auto callable = extract_callable_or_throw(ch["eta_table"], path + ".eta_table");
          channel.eta_table = to_config_callable(callable);
          register_callable(path + ".eta_table", callable, ch["eta_table"]);
        }
        if (has_key(ch, "eval_nc_fraction")) {
          channel.eval_nc_fraction =
              numeric_as_double(ch["eval_nc_fraction"], path + ".eval_nc_fraction");
        }
        if (has_key(ch, "threshold_multiplier")) {
          channel.threshold_multiplier =
              numeric_as_double(ch["threshold_multiplier"], path + ".threshold_multiplier");
        }
        if (has_key(ch, "eta_inf")) {
          channel.eta_inf = numeric_as_double(ch["eta_inf"], path + ".eta_inf");
        }
        if (has_key(ch, "eta_hard_cap")) {
          channel.eta_hard_cap =
              numeric_as_double(ch["eta_hard_cap"], path + ".eta_hard_cap");
        }
        if (has_key(ch, "shape_coefficient")) {
          channel.shape_coefficient =
              numeric_as_double(ch["shape_coefficient"], path + ".shape_coefficient");
        }
        if (has_key(ch, "relaxation_model")) {
          channel.relaxation_model =
              strict_string(ch["relaxation_model"], path + ".relaxation_model");
        }
        if (has_key(ch, "relaxation_tau_s")) {
          channel.relaxation_tau_s =
              numeric_as_double(ch["relaxation_tau_s"], path + ".relaxation_tau_s");
        }
        if (has_key(ch, "relaxation_tau_min_s")) {
          channel.relaxation_tau_min_s =
              numeric_as_double(ch["relaxation_tau_min_s"],
                                path + ".relaxation_tau_min_s");
        }
        if (has_key(ch, "relaxation_tau_max_s")) {
          channel.relaxation_tau_max_s =
              numeric_as_double(ch["relaxation_tau_max_s"],
                                path + ".relaxation_tau_max_s");
        }
        if (laser.hot_electron.eta_mode == "model") {
          if (channel.eval_nc_fraction < 0.0) {
            channel.eval_nc_fraction = channel.capture_nc_fraction;
          }
          if (channel.mechanism == "tpd") {
            if (channel.threshold_multiplier < 0.0) {
              channel.threshold_multiplier = 1.0;
            }
            if (channel.eta_inf < 0.0) {
              channel.eta_inf = 0.01;
            }
            if (channel.eta_hard_cap < 0.0) {
              channel.eta_hard_cap = 0.03;
            }
            if (channel.relaxation_model.empty()) {
              channel.relaxation_model = "vu2012";
            }
          } else if (channel.mechanism == "srs") {
            if (channel.threshold_multiplier < 0.0) {
              channel.threshold_multiplier = 8.0;
            }
            if (channel.eta_inf < 0.0) {
              channel.eta_inf = 0.08;
            }
            if (channel.eta_hard_cap < 0.0) {
              channel.eta_hard_cap = 0.08;
            }
            if (channel.relaxation_model.empty()) {
              channel.relaxation_model = "fixed";
            }
          }
        }
        if (has_key(ch, "T_hot_eV")) {
          channel.T_hot_eV = numeric_as_double(ch["T_hot_eV"], path + ".T_hot_eV");
        }
        if (!(channel.T_hot_eV > 0.0)) {
          throw ValueError(path + ".T_hot_eV must be > 0");
        }
        if (has_key(ch, "n_energy_groups")) {
          channel.n_energy_groups = strict_int32(ch["n_energy_groups"], path + ".n_energy_groups");
        }
        if (channel.n_energy_groups < 1) {
          throw ValueError(path + ".n_energy_groups must be >= 1");
        }
        if (has_key(ch, "E_min_over_Th")) {
          channel.E_min_over_Th = numeric_as_double(ch["E_min_over_Th"], path + ".E_min_over_Th");
        }
        if (!(channel.E_min_over_Th > 0.0)) {
          throw ValueError(path + ".E_min_over_Th must be > 0");
        }
        if (has_key(ch, "E_max_over_Th")) {
          channel.E_max_over_Th = numeric_as_double(ch["E_max_over_Th"], path + ".E_max_over_Th");
        }
        if (!(channel.E_max_over_Th > channel.E_min_over_Th)) {
          throw ValueError(path + ".E_max_over_Th must be > E_min_over_Th");
        }
        if (has_key(ch, "theta_div_deg")) {
          channel.theta_div_deg = numeric_as_double(ch["theta_div_deg"], path + ".theta_div_deg");
          if (channel.theta_div_deg < 0.0 || channel.theta_div_deg > 90.0) {
            throw ValueError(path + ".theta_div_deg must be in [0, 90]");
          }
        }
        if (has_key(ch, "tpd_theta_deg")) {
          channel.tpd_theta_deg = numeric_as_double(ch["tpd_theta_deg"], path + ".tpd_theta_deg");
        }
        if (channel.mechanism == "tpd" &&
            (!(channel.tpd_theta_deg > 0.0) || channel.tpd_theta_deg > 90.0)) {
          throw ValueError(path + ".tpd_theta_deg must be in (0, 90]");
        }
        if (has_key(ch, "tpd_delta_deg")) {
          channel.tpd_delta_deg = numeric_as_double(ch["tpd_delta_deg"], path + ".tpd_delta_deg");
        }
        if (channel.mechanism == "tpd" &&
            (channel.tpd_delta_deg < 0.0 || channel.tpd_delta_deg > 90.0)) {
          throw ValueError(path + ".tpd_delta_deg must be in [0, 90]");
        }
        if (has_key(ch, "n_mu")) {
          channel.n_mu = strict_int32(ch["n_mu"], path + ".n_mu");
          if (channel.n_mu < 1) {
            throw ValueError(path + ".n_mu must be >= 1");
          }
        }
        if (has_key(ch, "n_phi")) {
          channel.n_phi = strict_int32(ch["n_phi"], path + ".n_phi");
          if (channel.n_phi < 1) {
            throw ValueError(path + ".n_phi must be >= 1");
          }
        }
        laser.hot_electron.sources.push_back(std::move(channel));
      }
    }
  }

  if (has_key(kwargs, "profile")) {
    const py::handle profile_obj = kwargs["profile"];
    if (!py::isinstance<py::dict>(profile_obj)) {
      throw_value_type_error("Laser.profile", "dict", profile_obj);
    }
    const py::dict profile = py::reinterpret_borrow<py::dict>(profile_obj);
    enforce_known_keys(profile, "Laser.profile", {"model", "w0_um", "m", "r_um", "I_rel"});
    if (has_key(profile, "model")) {
      laser.profile_model = strict_string(profile["model"], "Laser.profile.model");
    }
    if (has_key(profile, "w0_um")) {
      laser.profile_w0_um = numeric_as_double(profile["w0_um"], "Laser.profile.w0_um");
    }
    if (has_key(profile, "m")) {
      laser.profile_m = strict_int32(profile["m"], "Laser.profile.m");
    }
    const bool is_table = laser.profile_model == "table";
    if (is_table) {
      if (!has_key(profile, "r_um") || !has_key(profile, "I_rel")) {
        throw ConfigError("Laser profile model=\"table\" requires r_um and I_rel");
      }
      if (has_key(profile, "w0_um") || has_key(profile, "m")) {
        throw ConfigError("Laser profile model=\"table\" does not take w0_um/m");
      }
      const std::vector<double> r_um_v =
          strict_double_vector(profile["r_um"], "Laser.profile.r_um");
      const std::vector<double> I_v =
          strict_double_vector(profile["I_rel"], "Laser.profile.I_rel");
      if (r_um_v.size() < 2 || r_um_v.size() != I_v.size()) {
        throw ConfigError(
            "Laser profile table needs >= 2 points with equal-length r_um and I_rel");
      }
      double prev = -1.0;
      double imax = 0.0;
      for (std::size_t k = 0; k < r_um_v.size(); ++k) {
        if (!(r_um_v[k] >= 0.0) || !(r_um_v[k] > prev)) {
          throw ConfigError("Laser profile table r_um must be >= 0 and strictly ascending");
        }
        prev = r_um_v[k];
        if (!(I_v[k] >= 0.0)) {
          throw ConfigError("Laser profile table I_rel must be >= 0");
        }
        imax = std::max(imax, I_v[k]);
      }
      if (!(imax > 0.0)) {
        throw ConfigError("Laser profile table I_rel must not be all zero");
      }
      laser.profile_r_cm.clear();
      for (const double r : r_um_v) laser.profile_r_cm.push_back(r * 1.0e-4);
      laser.profile_I = I_v;
    } else if (has_key(profile, "r_um") || has_key(profile, "I_rel")) {
      throw ConfigError("Laser profile r_um/I_rel require model=\"table\"");
    }
  }

  if (!has_key(kwargs, "beams")) {
    return;
  }

  const py::handle beams_obj = kwargs["beams"];
  if (!py::isinstance<py::sequence>(beams_obj) || py::isinstance<py::str>(beams_obj)) {
    throw_value_type_error("Laser.beams", "list[dict]", beams_obj);
  }
  const py::sequence beams = py::reinterpret_borrow<py::sequence>(beams_obj);
  laser.beams.clear();
  laser.beams.reserve(beams.size());
  for (std::size_t i = 0; i < beams.size(); ++i) {
    const py::handle beam_obj = beams[i];
    if (!py::isinstance<py::dict>(beam_obj)) {
      throw_value_type_error("Laser.beams[" + std::to_string(i) + "]", "dict", beam_obj);
    }
    py::dict beam = py::reinterpret_borrow<py::dict>(beam_obj);
    enforce_known_keys(beam, "Laser.beams",
                       {"name", "direction", "theta", "phi", "f_number", "focus",
                        "defocus_DR", "delta_lambda_nm", "power", "profile", "profile_model",
                        "profile_w0_um", "profile_m", "spot"});

    if (has_key(beam, "spot")) {
      if (has_key(beam, "profile")) {
        throw ConfigError("Laser.beams[" + std::to_string(i) +
                          "] cannot specify both spot and profile");
      } else {
        const py::handle spot_obj = beam["spot"];
        if (!py::isinstance<py::dict>(spot_obj)) {
          throw_value_type_error("Laser.beams.spot", "dict", spot_obj);
        }
        const py::dict spot = py::reinterpret_borrow<py::dict>(spot_obj);
        enforce_known_keys(spot, "Laser.beams.spot", {"model", "radius_um", "m"});

        py::dict profile;
        if (has_key(spot, "model")) {
          profile["model"] = spot["model"];
        }
        if (has_key(spot, "radius_um")) {
          profile["w0_um"] = spot["radius_um"];
        }
        if (has_key(spot, "m")) {
          profile["m"] = spot["m"];
        }
        beam["profile"] = std::move(profile);
        tenryu::core::log_warning("LaserBeam.spot is deprecated; converted to profile");
      }
    }

    Config::LaserConfig::BeamDef out;
    if (has_key(beam, "name")) {
      out.name = strict_string(beam["name"], "Laser.beams[" + std::to_string(i) + "].name");
    }
    if (has_key(beam, "direction")) {
      out.direction = strict_double_vector(
          beam["direction"], "Laser.beams[" + std::to_string(i) + "].direction");
      if (out.direction.size() != 3) {
        throw ValueError("Laser.beams[" + std::to_string(i) +
                         "].direction must have exactly 3 elements");
      }
    }
    if (has_key(beam, "theta")) {
      out.theta =
          numeric_as_double(beam["theta"], "Laser.beams[" + std::to_string(i) + "].theta");
    }
    if (has_key(beam, "phi")) {
      out.phi =
          numeric_as_double(beam["phi"], "Laser.beams[" + std::to_string(i) + "].phi");
    }
    if (has_key(beam, "f_number")) {
      out.f_number = numeric_as_double(
          beam["f_number"], "Laser.beams[" + std::to_string(i) + "].f_number");
      ensure_positive(out.f_number, "Laser.beams.f_number");
    }
    if (has_key(beam, "focus")) {
      out.focus =
          strict_double_vector(beam["focus"], "Laser.beams[" + std::to_string(i) + "].focus");
      if (out.focus.size() != 3) {
        throw ValueError("Laser.beams[" + std::to_string(i) +
                         "].focus must have exactly 3 elements");
      }
    }
    if (has_key(beam, "defocus_DR")) {
      // Keep this mixed-style key for backward compatibility with existing inputs.
      out.defocus_DR = numeric_as_double(
          beam["defocus_DR"], "Laser.beams[" + std::to_string(i) + "].defocus_DR");
    }
    if (has_key(beam, "delta_lambda_nm")) {
      out.delta_lambda_nm = numeric_as_double(
          beam["delta_lambda_nm"], "Laser.beams[" + std::to_string(i) + "].delta_lambda_nm");
    }

    if (has_key(beam, "profile")) {
      const py::handle profile_obj = beam["profile"];
      if (!py::isinstance<py::dict>(profile_obj)) {
        throw_value_type_error("Laser.beams.profile", "dict", profile_obj);
      }
      const py::dict profile = py::reinterpret_borrow<py::dict>(profile_obj);
      enforce_known_keys(profile, "Laser.beams.profile",
                         {"model", "w0_um", "m", "r_um", "I_rel"});
      if (has_key(profile, "model")) {
        out.profile_model = strict_string(profile["model"], "Laser.beams.profile.model");
      }
      if (has_key(profile, "w0_um")) {
        out.profile_w0_um =
            numeric_as_double(profile["w0_um"], "Laser.beams.profile.w0_um");
      }
      if (has_key(profile, "m")) {
        out.profile_m = strict_int32(profile["m"], "Laser.beams.profile.m");
      }
      const bool is_table = out.profile_model == "table";
      if (is_table) {
        if (!has_key(profile, "r_um") || !has_key(profile, "I_rel")) {
          throw ConfigError("Laser profile model=\"table\" requires r_um and I_rel");
        }
        if (has_key(profile, "w0_um") || has_key(profile, "m")) {
          throw ConfigError("Laser profile model=\"table\" does not take w0_um/m");
        }
        const std::vector<double> r_um_v =
            strict_double_vector(profile["r_um"], "Laser.beams.profile.r_um");
        const std::vector<double> I_v =
            strict_double_vector(profile["I_rel"], "Laser.beams.profile.I_rel");
        if (r_um_v.size() < 2 || r_um_v.size() != I_v.size()) {
          throw ConfigError(
              "Laser profile table needs >= 2 points with equal-length r_um and I_rel");
        }
        double prev = -1.0;
        double imax = 0.0;
        for (std::size_t k = 0; k < r_um_v.size(); ++k) {
          if (!(r_um_v[k] >= 0.0) || !(r_um_v[k] > prev)) {
            throw ConfigError("Laser profile table r_um must be >= 0 and strictly ascending");
          }
          prev = r_um_v[k];
          if (!(I_v[k] >= 0.0)) {
            throw ConfigError("Laser profile table I_rel must be >= 0");
          }
          imax = std::max(imax, I_v[k]);
        }
        if (!(imax > 0.0)) {
          throw ConfigError("Laser profile table I_rel must not be all zero");
        }
        out.profile_r_cm.clear();
        for (const double r : r_um_v) out.profile_r_cm.push_back(r * 1.0e-4);
        out.profile_I = I_v;
      } else if (has_key(profile, "r_um") || has_key(profile, "I_rel")) {
        throw ConfigError("Laser profile r_um/I_rel require model=\"table\"");
      }
    }
    if (has_key(beam, "profile_model")) {
      out.profile_model =
          strict_string(beam["profile_model"], "Laser.beams.profile_model");
    }
    if (has_key(beam, "profile_w0_um")) {
      out.profile_w0_um =
          numeric_as_double(beam["profile_w0_um"], "Laser.beams.profile_w0_um");
    }
    if (has_key(beam, "profile_m")) {
      out.profile_m =
          strict_int32(beam["profile_m"], "Laser.beams.profile_m");
    }

    if (has_key(beam, "power")) {
      const auto callable =
          extract_callable_or_throw(beam["power"],
                                    "Laser.beams[" + std::to_string(i) + "].power");
      out.power = to_config_callable(callable);
      register_callable("Laser.beams[" + std::to_string(i) + "].power", callable,
                        beam["power"]);
    }

    laser.beams.push_back(std::move(out));
  }
}

void Builder::set_numerics(py::dict kwargs) {
  mark_block_called(Block::Numerics);
  enforce_known_keys(kwargs, "Numerics",
                     {"splitting_order", "splitting", "T_start_eV", "coulomb_log_floor",
                      "has_physical_rz_axis",
                      "persistent_loop", "z_reflection", "dt", "debug", "hydro", "conduction", "ale",
                      "plic", "materials", "ale1d",
                      "floors", "positivity", "safety", "cell_search", "diagnostics",
                      "diagnostics_every", "profile", "cfl",
                      "radiation_thermal_subcycle"});

  auto& numerics = config.numerics;
  if (has_key(kwargs, "splitting_order")) {
    warn_ignored_key("Numerics.splitting_order");
  }
  if (has_key(kwargs, "splitting")) {
    warn_ignored_key("Numerics.splitting");
  }
  if (has_key(kwargs, "coulomb_log_floor")) {
    warn_ignored_key("Numerics.coulomb_log_floor");
  }
  if (has_key(kwargs, "cell_search")) {
    warn_ignored_key("Numerics.cell_search");
  }
  if (has_key(kwargs, "diagnostics_every")) {
    numerics.diagnostics_every = strict_int32(
        kwargs["diagnostics_every"], "Numerics.diagnostics_every");
    ensure_int_ge(numerics.diagnostics_every, 1, "Numerics.diagnostics_every");
  }
  if (has_key(kwargs, "cfl")) {
    const double cfl = numeric_as_double(kwargs["cfl"], "Numerics.cfl");
    if (!(cfl > 0.0 && cfl <= 1.0)) {
      throw ConfigError("Numerics.cfl must be in (0, 1]");
    }
    numerics.dt.cfl_hydro = cfl;
    numerics.dt.cfl_cond = cfl;
    tenryu::core::log_warning(
        "Numerics.cfl is deprecated; mapped to Numerics.dt.cfl_hydro and Numerics.dt.cfl_cond");
  }
  if (has_key(kwargs, "T_start_eV")) {
    hydro_t_start_eV = numeric_as_double(kwargs["T_start_eV"], "Numerics.T_start_eV");
    ensure_non_negative(hydro_t_start_eV, "Numerics.T_start_eV");
    numerics.T_start_eV = hydro_t_start_eV;
    tenryu::core::log_warning("Numerics.T_start_eV is deprecated; use Numerics.hydro.T_start_eV");
  }
  if (has_key(kwargs, "radiation_thermal_subcycle")) {
    numerics.radiation_thermal_subcycle = strict_bool(
        kwargs["radiation_thermal_subcycle"],
        "Numerics.radiation_thermal_subcycle");
  }
  if (has_key(kwargs, "has_physical_rz_axis")) {
    (void)strict_bool(kwargs["has_physical_rz_axis"],
                      "Numerics.has_physical_rz_axis");
  }
  if (has_key(kwargs, "persistent_loop")) {
    const py::handle persistent_loop_obj = kwargs["persistent_loop"];
    if (!py::isinstance<py::dict>(persistent_loop_obj)) {
      throw_value_type_error("Numerics.persistent_loop", "dict", persistent_loop_obj);
    }
    const py::dict persistent_loop =
        py::reinterpret_borrow<py::dict>(persistent_loop_obj);
    enforce_known_keys(persistent_loop, "Numerics.persistent_loop",
                       {"enabled", "chunk_steps"});
    if (has_key(persistent_loop, "enabled")) {
      numerics.persistent_loop.enabled =
          strict_bool(persistent_loop["enabled"], "Numerics.persistent_loop.enabled");
    }
    if (has_key(persistent_loop, "chunk_steps")) {
      numerics.persistent_loop.chunk_steps = strict_int32(
          persistent_loop["chunk_steps"], "Numerics.persistent_loop.chunk_steps");
    }
  }
  if (has_key(kwargs, "z_reflection")) {
    const py::handle z_reflection_obj = kwargs["z_reflection"];
    if (!py::isinstance<py::dict>(z_reflection_obj)) {
      throw_value_type_error(
          "Numerics.z_reflection", "dict", z_reflection_obj);
    }
    const py::dict z_reflection =
        py::reinterpret_borrow<py::dict>(z_reflection_obj);
    enforce_known_keys(z_reflection, "Numerics.z_reflection", {"mode"});
    if (has_key(z_reflection, "mode")) {
      numerics.z_reflection.mode = strict_string(
          z_reflection["mode"], "Numerics.z_reflection.mode");
    }
  }
  if (has_key(kwargs, "debug")) {
    const py::handle debug_obj = kwargs["debug"];
    if (!py::isinstance<py::dict>(debug_obj)) {
      throw_value_type_error("Numerics.debug", "dict", debug_obj);
    }
    const py::dict debug = py::reinterpret_borrow<py::dict>(debug_obj);
    enforce_known_keys(debug, "Numerics.debug",
                       {"trace_mesh_motion",
                        "trace_mesh_node_selector",
                        "trace_mesh_cell",
                        "trace_max_steps"});
    if (has_key(debug, "trace_mesh_motion")) {
      numerics.debug.trace_mesh_motion =
          strict_bool(debug["trace_mesh_motion"],
                      "Numerics.debug.trace_mesh_motion");
    }
    if (has_key(debug, "trace_mesh_node_selector")) {
      numerics.debug.trace_mesh_node_selector =
          strict_string(debug["trace_mesh_node_selector"],
                        "Numerics.debug.trace_mesh_node_selector");
      if (numerics.debug.trace_mesh_node_selector != "outer_equator") {
        throw ValueError(
            "Numerics.debug.trace_mesh_node_selector must be \"outer_equator\"");
      }
    }
    if (has_key(debug, "trace_mesh_cell")) {
      numerics.debug.trace_mesh_cell =
          strict_int32(debug["trace_mesh_cell"],
                       "Numerics.debug.trace_mesh_cell");
      ensure_non_negative(numerics.debug.trace_mesh_cell,
                          "Numerics.debug.trace_mesh_cell");
    }
    if (has_key(debug, "trace_max_steps")) {
      numerics.debug.trace_max_steps =
          strict_int32(debug["trace_max_steps"],
                       "Numerics.debug.trace_max_steps");
      ensure_non_negative(numerics.debug.trace_max_steps,
                          "Numerics.debug.trace_max_steps");
    }
  }
  if (has_key(kwargs, "diagnostics")) {
    const py::handle diagnostics_obj = kwargs["diagnostics"];
    if (!py::isinstance<py::dict>(diagnostics_obj)) {
      throw_value_type_error("Numerics.diagnostics", "dict", diagnostics_obj);
    }
    const py::dict diagnostics = py::reinterpret_borrow<py::dict>(diagnostics_obj);
    enforce_known_keys(diagnostics, "Numerics.diagnostics",
                       {"phase_resolved_energy",
                        "r_momentum_source_audit",
                        "dt_breakdown_history_enabled",
                        "mesh_attribution",
                        "icf",
                        "hotspot_gas",
                        "conservation",
                        "refinement_estimator",
                        "refinement_autopilot",
                        "evacuated_cell_shadow",
                        "ale_provenance_emission",
                        "conduction_energy_rate_export",
                        "mesh_quality_min",
                        "shock_approach",
                        "ale_velcoherence",
                        "production_audit",
                        "mesh_degeneracy_forensics"});
    if (has_key(diagnostics, "phase_resolved_energy")) {
      numerics.diagnostics.phase_resolved_energy = strict_bool(
          diagnostics["phase_resolved_energy"],
          "Numerics.diagnostics.phase_resolved_energy");
    }
    if (has_key(diagnostics, "r_momentum_source_audit")) {
      numerics.diagnostics.r_momentum_source_audit = strict_bool(
          diagnostics["r_momentum_source_audit"],
          "Numerics.diagnostics.r_momentum_source_audit");
    }
    if (has_key(diagnostics, "dt_breakdown_history_enabled")) {
      numerics.diagnostics.dt_breakdown_history_enabled = strict_bool(
          diagnostics["dt_breakdown_history_enabled"],
          "Numerics.diagnostics.dt_breakdown_history_enabled");
    }
    if (has_key(diagnostics, "mesh_attribution")) {
      const py::handle attr_obj = diagnostics["mesh_attribution"];
      if (!py::isinstance<py::dict>(attr_obj)) {
        throw_value_type_error("Numerics.diagnostics.mesh_attribution",
                               "dict",
                               attr_obj);
      }
      const py::dict attr = py::reinterpret_borrow<py::dict>(attr_obj);
      enforce_known_keys(attr,
                         "Numerics.diagnostics.mesh_attribution",
                         {"enabled",
                          "record_node_displacements",
                          "dump_on_failure_only",
                          "enable_leave_one_out_replay"});
      auto& cfg_attr = numerics.diagnostics.mesh_attribution;
      if (has_key(attr, "enabled")) {
        cfg_attr.enabled = strict_bool(attr["enabled"],
                                      "Numerics.diagnostics.mesh_attribution.enabled");
      }
      if (has_key(attr, "record_node_displacements")) {
        cfg_attr.record_node_displacements = strict_bool(
            attr["record_node_displacements"],
            "Numerics.diagnostics.mesh_attribution.record_node_displacements");
      }
      if (has_key(attr, "dump_on_failure_only")) {
        cfg_attr.dump_on_failure_only = strict_bool(
            attr["dump_on_failure_only"],
            "Numerics.diagnostics.mesh_attribution.dump_on_failure_only");
      }
      if (has_key(attr, "enable_leave_one_out_replay")) {
        cfg_attr.enable_leave_one_out_replay = strict_bool(
            attr["enable_leave_one_out_replay"],
            "Numerics.diagnostics.mesh_attribution.enable_leave_one_out_replay");
      }
    }
    if (has_key(diagnostics, "icf")) {
      const py::handle icf_obj = diagnostics["icf"];
      if (!py::isinstance<py::dict>(icf_obj)) {
        throw_value_type_error("Numerics.diagnostics.icf", "dict", icf_obj);
      }
      const py::dict icf = py::reinterpret_borrow<py::dict>(icf_obj);
      enforce_known_keys(icf,
                         "Numerics.diagnostics.icf",
                         {"enabled",
                          "rho_inner_threshold_g_per_cc",
                          "rho_outer_threshold_g_per_cc"});
      auto& cfg_icf = numerics.diagnostics.icf;
      if (has_key(icf, "enabled")) {
        cfg_icf.enabled = strict_bool(icf["enabled"],
                                      "Numerics.diagnostics.icf.enabled");
      }
      if (has_key(icf, "rho_inner_threshold_g_per_cc")) {
        cfg_icf.rho_inner_threshold_g_per_cc =
            numeric_as_double(icf["rho_inner_threshold_g_per_cc"],
                              "Numerics.diagnostics.icf."
                              "rho_inner_threshold_g_per_cc");
      }
      if (has_key(icf, "rho_outer_threshold_g_per_cc")) {
        cfg_icf.rho_outer_threshold_g_per_cc =
            numeric_as_double(icf["rho_outer_threshold_g_per_cc"],
                              "Numerics.diagnostics.icf."
                              "rho_outer_threshold_g_per_cc");
      }
    }
    if (has_key(diagnostics, "hotspot_gas")) {
      const py::handle hotspot_obj = diagnostics["hotspot_gas"];
      if (!py::isinstance<py::dict>(hotspot_obj)) {
        throw_value_type_error("Numerics.diagnostics.hotspot_gas",
                               "dict",
                               hotspot_obj);
      }
      const py::dict hotspot = py::reinterpret_borrow<py::dict>(hotspot_obj);
      enforce_known_keys(hotspot,
                         "Numerics.diagnostics.hotspot_gas",
                         {"enabled", "R_g_cm", "mass_drift_warn_rel"});
      auto& cfg_hotspot = numerics.diagnostics.hotspot_gas;
      if (has_key(hotspot, "enabled")) {
        cfg_hotspot.enabled = strict_bool(
            hotspot["enabled"], "Numerics.diagnostics.hotspot_gas.enabled");
      }
      if (has_key(hotspot, "R_g_cm")) {
        cfg_hotspot.R_g_cm = numeric_as_double(
            hotspot["R_g_cm"], "Numerics.diagnostics.hotspot_gas.R_g_cm");
      }
      if (has_key(hotspot, "mass_drift_warn_rel")) {
        cfg_hotspot.mass_drift_warn_rel = numeric_as_double(
            hotspot["mass_drift_warn_rel"],
            "Numerics.diagnostics.hotspot_gas.mass_drift_warn_rel");
      }
    }
    if (has_key(diagnostics, "conservation")) {
      const py::handle conservation_obj = diagnostics["conservation"];
      if (!py::isinstance<py::dict>(conservation_obj)) {
        throw_value_type_error("Numerics.diagnostics.conservation",
                               "dict",
                               conservation_obj);
      }
      const py::dict conservation =
          py::reinterpret_borrow<py::dict>(conservation_obj);
      enforce_known_keys(conservation,
                         "Numerics.diagnostics.conservation",
                         {"enabled"});
      if (has_key(conservation, "enabled")) {
        numerics.diagnostics.conservation.enabled =
            strict_bool(conservation["enabled"],
                        "Numerics.diagnostics.conservation.enabled");
      }
    }
    if (has_key(diagnostics, "refinement_estimator")) {
      const py::handle estimator_obj = diagnostics["refinement_estimator"];
      if (!py::isinstance<py::dict>(estimator_obj)) {
        throw_value_type_error("Numerics.diagnostics.refinement_estimator",
                               "dict",
                               estimator_obj);
      }
      const py::dict estimator =
          py::reinterpret_borrow<py::dict>(estimator_obj);
      enforce_known_keys(estimator,
                         "Numerics.diagnostics.refinement_estimator",
                         {"enabled", "every", "filter_eps", "detect_cutoff"});
      auto& cfg_estimator = numerics.diagnostics.refinement_estimator;
      if (has_key(estimator, "enabled")) {
        cfg_estimator.enabled = strict_bool(
            estimator["enabled"],
            "Numerics.diagnostics.refinement_estimator.enabled");
      }
      if (has_key(estimator, "every")) {
        cfg_estimator.every = strict_int32(
            estimator["every"],
            "Numerics.diagnostics.refinement_estimator.every");
        ensure_int_ge(cfg_estimator.every,
                      1,
                      "Numerics.diagnostics.refinement_estimator.every");
      }
      if (has_key(estimator, "filter_eps")) {
        cfg_estimator.filter_eps = numeric_as_double(
            estimator["filter_eps"],
            "Numerics.diagnostics.refinement_estimator.filter_eps");
        if (!(std::isfinite(cfg_estimator.filter_eps) &&
              cfg_estimator.filter_eps > 0.0)) {
          throw ValueError(
              "Numerics.diagnostics.refinement_estimator.filter_eps must be "
              "finite and > 0");
        }
      }
      if (has_key(estimator, "detect_cutoff")) {
        cfg_estimator.detect_cutoff = numeric_as_double(
            estimator["detect_cutoff"],
            "Numerics.diagnostics.refinement_estimator.detect_cutoff");
        if (!(std::isfinite(cfg_estimator.detect_cutoff) &&
              cfg_estimator.detect_cutoff > 0.0 &&
              cfg_estimator.detect_cutoff < 1.0)) {
          throw ValueError(
              "Numerics.diagnostics.refinement_estimator.detect_cutoff must "
              "be finite and in (0, 1)");
        }
      }
    }
    if (has_key(diagnostics, "refinement_autopilot")) {
      const py::handle autopilot_obj = diagnostics["refinement_autopilot"];
      if (!py::isinstance<py::dict>(autopilot_obj)) {
        throw_value_type_error("Numerics.diagnostics.refinement_autopilot",
                               "dict",
                               autopilot_obj);
      }
      const py::dict autopilot =
          py::reinterpret_borrow<py::dict>(autopilot_obj);
      enforce_known_keys(autopilot,
                         "Numerics.diagnostics.refinement_autopilot",
                         {"enabled",
                          "mode",
                          "ckpt_lead_h",
                          "e_on",
                          "e_off",
                          "assoc_cut",
                          "strong_cut",
                          "gap_bridge",
                          "persist",
                          "n_q_plan",
                          "chi_design",
                          "s_rep_cm",
                          "handoff_cm",
                          "window_lo_h",
                          "window_hi_h",
                          "history",
                          "cov_min"});
      auto& cfg_autopilot = numerics.diagnostics.refinement_autopilot;
      if (has_key(autopilot, "enabled")) {
        cfg_autopilot.enabled = strict_bool(
            autopilot["enabled"],
            "Numerics.diagnostics.refinement_autopilot.enabled");
      }
      if (has_key(autopilot, "mode")) {
        cfg_autopilot.mode = strict_string(
            autopilot["mode"],
            "Numerics.diagnostics.refinement_autopilot.mode");
      }
      if (has_key(autopilot, "ckpt_lead_h")) {
        cfg_autopilot.ckpt_lead_h = numeric_as_double(
            autopilot["ckpt_lead_h"],
            "Numerics.diagnostics.refinement_autopilot.ckpt_lead_h");
      }
      if (has_key(autopilot, "e_on")) {
        cfg_autopilot.e_on = numeric_as_double(
            autopilot["e_on"],
            "Numerics.diagnostics.refinement_autopilot.e_on");
      }
      if (has_key(autopilot, "e_off")) {
        cfg_autopilot.e_off = numeric_as_double(
            autopilot["e_off"],
            "Numerics.diagnostics.refinement_autopilot.e_off");
      }
      if (has_key(autopilot, "assoc_cut")) {
        cfg_autopilot.assoc_cut = numeric_as_double(
            autopilot["assoc_cut"],
            "Numerics.diagnostics.refinement_autopilot.assoc_cut");
      }
      if (has_key(autopilot, "strong_cut")) {
        cfg_autopilot.strong_cut = numeric_as_double(
            autopilot["strong_cut"],
            "Numerics.diagnostics.refinement_autopilot.strong_cut");
      }
      if (has_key(autopilot, "gap_bridge")) {
        cfg_autopilot.gap_bridge = strict_int32(
            autopilot["gap_bridge"],
            "Numerics.diagnostics.refinement_autopilot.gap_bridge");
      }
      if (has_key(autopilot, "persist")) {
        cfg_autopilot.persist = strict_int32(
            autopilot["persist"],
            "Numerics.diagnostics.refinement_autopilot.persist");
      }
      if (has_key(autopilot, "n_q_plan")) {
        cfg_autopilot.n_q_plan = numeric_as_double(
            autopilot["n_q_plan"],
            "Numerics.diagnostics.refinement_autopilot.n_q_plan");
      }
      if (has_key(autopilot, "chi_design")) {
        cfg_autopilot.chi_design = numeric_as_double(
            autopilot["chi_design"],
            "Numerics.diagnostics.refinement_autopilot.chi_design");
      }
      if (has_key(autopilot, "s_rep_cm")) {
        cfg_autopilot.s_rep_cm = numeric_as_double(
            autopilot["s_rep_cm"],
            "Numerics.diagnostics.refinement_autopilot.s_rep_cm");
      }
      if (has_key(autopilot, "handoff_cm")) {
        cfg_autopilot.handoff_cm = numeric_as_double(
            autopilot["handoff_cm"],
            "Numerics.diagnostics.refinement_autopilot.handoff_cm");
      }
      if (has_key(autopilot, "window_lo_h")) {
        cfg_autopilot.window_lo_h = numeric_as_double(
            autopilot["window_lo_h"],
            "Numerics.diagnostics.refinement_autopilot.window_lo_h");
      }
      if (has_key(autopilot, "window_hi_h")) {
        cfg_autopilot.window_hi_h = numeric_as_double(
            autopilot["window_hi_h"],
            "Numerics.diagnostics.refinement_autopilot.window_hi_h");
      }
      if (has_key(autopilot, "history")) {
        cfg_autopilot.history = strict_int32(
            autopilot["history"],
            "Numerics.diagnostics.refinement_autopilot.history");
      }
      if (has_key(autopilot, "cov_min")) {
        cfg_autopilot.cov_min = numeric_as_double(
            autopilot["cov_min"],
            "Numerics.diagnostics.refinement_autopilot.cov_min");
      }

      const auto in_open_unit_interval = [](const double value) {
        return std::isfinite(value) && value > 0.0 && value < 1.0;
      };
      if (cfg_autopilot.mode != "shadow" &&
          cfg_autopilot.mode != "arm_exit") {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.mode must be exactly "
            "\"shadow\" or \"arm_exit\"");
      }
      if (!(std::isfinite(cfg_autopilot.ckpt_lead_h) &&
            cfg_autopilot.ckpt_lead_h > 0.0)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.ckpt_lead_h must be "
            "finite and > 0");
      }
      if (!in_open_unit_interval(cfg_autopilot.e_on)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.e_on must be finite "
            "and in (0, 1)");
      }
      if (!in_open_unit_interval(cfg_autopilot.e_off)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.e_off must be finite "
            "and in (0, 1)");
      }
      if (!in_open_unit_interval(cfg_autopilot.assoc_cut)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.assoc_cut must be "
            "finite and in (0, 1)");
      }
      if (!in_open_unit_interval(cfg_autopilot.strong_cut)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.strong_cut must be "
            "finite and in (0, 1)");
      }
      if (!(cfg_autopilot.e_off < cfg_autopilot.e_on)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.e_off must be < e_on");
      }
      if (!(cfg_autopilot.assoc_cut <= cfg_autopilot.strong_cut)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.assoc_cut must be <= "
            "strong_cut");
      }
      ensure_non_negative(
          cfg_autopilot.gap_bridge,
          "Numerics.diagnostics.refinement_autopilot.gap_bridge");
      ensure_int_ge(cfg_autopilot.persist,
                    1,
                    "Numerics.diagnostics.refinement_autopilot.persist");
      if (!(std::isfinite(cfg_autopilot.n_q_plan) &&
            cfg_autopilot.n_q_plan > 0.0)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.n_q_plan must be "
            "finite and > 0");
      }
      if (!in_open_unit_interval(cfg_autopilot.chi_design)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.chi_design must be "
            "finite and in (0, 1)");
      }
      if (!(std::isfinite(cfg_autopilot.s_rep_cm) &&
            std::isfinite(cfg_autopilot.handoff_cm) &&
            cfg_autopilot.s_rep_cm > cfg_autopilot.handoff_cm &&
            cfg_autopilot.handoff_cm > 0.0)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot requires s_rep_cm > "
            "handoff_cm > 0 with finite values");
      }
      if (!(std::isfinite(cfg_autopilot.window_lo_h) &&
            std::isfinite(cfg_autopilot.window_hi_h) &&
            cfg_autopilot.window_hi_h >= cfg_autopilot.window_lo_h &&
            cfg_autopilot.window_lo_h > 0.0)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot requires window_hi_h "
            ">= window_lo_h > 0 with finite values");
      }
      ensure_int_ge(cfg_autopilot.history,
                    3,
                    "Numerics.diagnostics.refinement_autopilot.history");
      if (!(std::isfinite(cfg_autopilot.cov_min) &&
            cfg_autopilot.cov_min > 0.0 && cfg_autopilot.cov_min <= 1.0)) {
        throw ValueError(
            "Numerics.diagnostics.refinement_autopilot.cov_min must be finite "
            "and in (0, 1]");
      }
    }
    if (has_key(diagnostics, "evacuated_cell_shadow")) {
      const py::handle shadow_obj = diagnostics["evacuated_cell_shadow"];
      if (!py::isinstance<py::dict>(shadow_obj)) {
        throw_value_type_error("Numerics.diagnostics.evacuated_cell_shadow",
                               "dict",
                               shadow_obj);
      }
      const py::dict shadow = py::reinterpret_borrow<py::dict>(shadow_obj);
      enforce_known_keys(shadow,
                         "Numerics.diagnostics.evacuated_cell_shadow",
                         {"enabled",
                          "every_n_steps",
                          "arm_mass_fraction",
                          "off_mass_fraction",
                          "rho_vacuum_policy_g_per_cc",
                          "laser_wavelength_nm",
                          "laser_ne_over_ncrit_max"});
      auto& cfg_shadow = numerics.diagnostics.evacuated_cell_shadow;
      if (has_key(shadow, "enabled")) {
        cfg_shadow.enabled = strict_bool(
            shadow["enabled"],
            "Numerics.diagnostics.evacuated_cell_shadow.enabled");
      }
      if (has_key(shadow, "every_n_steps")) {
        cfg_shadow.every_n_steps = strict_int32(
            shadow["every_n_steps"],
            "Numerics.diagnostics.evacuated_cell_shadow.every_n_steps");
      }
      if (has_key(shadow, "arm_mass_fraction")) {
        cfg_shadow.arm_mass_fraction = numeric_as_double(
            shadow["arm_mass_fraction"],
            "Numerics.diagnostics.evacuated_cell_shadow.arm_mass_fraction");
      }
      if (has_key(shadow, "off_mass_fraction")) {
        cfg_shadow.off_mass_fraction = numeric_as_double(
            shadow["off_mass_fraction"],
            "Numerics.diagnostics.evacuated_cell_shadow.off_mass_fraction");
      }
      if (has_key(shadow, "rho_vacuum_policy_g_per_cc")) {
        cfg_shadow.rho_vacuum_policy_g_per_cc = numeric_as_double(
            shadow["rho_vacuum_policy_g_per_cc"],
            "Numerics.diagnostics.evacuated_cell_shadow."
            "rho_vacuum_policy_g_per_cc");
      }
      if (has_key(shadow, "laser_wavelength_nm")) {
        cfg_shadow.laser_wavelength_nm = numeric_as_double(
            shadow["laser_wavelength_nm"],
            "Numerics.diagnostics.evacuated_cell_shadow."
            "laser_wavelength_nm");
      }
      if (has_key(shadow, "laser_ne_over_ncrit_max")) {
        cfg_shadow.laser_ne_over_ncrit_max = numeric_as_double(
            shadow["laser_ne_over_ncrit_max"],
            "Numerics.diagnostics.evacuated_cell_shadow."
            "laser_ne_over_ncrit_max");
      }

      ensure_int_ge(
          cfg_shadow.every_n_steps,
          1,
          "Numerics.diagnostics.evacuated_cell_shadow.every_n_steps");
      if (!(std::isfinite(cfg_shadow.off_mass_fraction) &&
            std::isfinite(cfg_shadow.arm_mass_fraction) &&
            cfg_shadow.off_mass_fraction > 0.0 &&
            cfg_shadow.off_mass_fraction < cfg_shadow.arm_mass_fraction &&
            cfg_shadow.arm_mass_fraction < 1.0)) {
        throw ValueError(
            "Numerics.diagnostics.evacuated_cell_shadow requires "
            "0 < off_mass_fraction < arm_mass_fraction < 1");
      }
      if (!(std::isfinite(cfg_shadow.rho_vacuum_policy_g_per_cc) &&
            cfg_shadow.rho_vacuum_policy_g_per_cc > 0.0)) {
        throw ValueError(
            "Numerics.diagnostics.evacuated_cell_shadow."
            "rho_vacuum_policy_g_per_cc must be finite and > 0");
      }
      if (!(std::isfinite(cfg_shadow.laser_wavelength_nm) &&
            cfg_shadow.laser_wavelength_nm > 0.0)) {
        throw ValueError(
            "Numerics.diagnostics.evacuated_cell_shadow."
            "laser_wavelength_nm must be finite and > 0");
      }
      if (!(std::isfinite(cfg_shadow.laser_ne_over_ncrit_max) &&
            cfg_shadow.laser_ne_over_ncrit_max > 0.0)) {
        throw ValueError(
            "Numerics.diagnostics.evacuated_cell_shadow."
            "laser_ne_over_ncrit_max must be finite and > 0");
      }
    }
    if (has_key(diagnostics, "ale_provenance_emission")) {
      const py::handle ale_prov_obj = diagnostics["ale_provenance_emission"];
      if (!py::isinstance<py::dict>(ale_prov_obj)) {
        throw_value_type_error("Numerics.diagnostics.ale_provenance_emission",
                               "dict",
                               ale_prov_obj);
      }
      const py::dict ale_prov =
          py::reinterpret_borrow<py::dict>(ale_prov_obj);
      enforce_known_keys(ale_prov,
                         "Numerics.diagnostics.ale_provenance_emission",
                         {"enabled"});
      if (has_key(ale_prov, "enabled")) {
        numerics.diagnostics.ale_provenance_emission.enabled =
            strict_bool(ale_prov["enabled"],
                        "Numerics.diagnostics.ale_provenance_emission.enabled");
      }
    }
    if (has_key(diagnostics, "conduction_energy_rate_export")) {
      const py::handle cond_rate_obj =
          diagnostics["conduction_energy_rate_export"];
      if (!py::isinstance<py::dict>(cond_rate_obj)) {
        throw_value_type_error(
            "Numerics.diagnostics.conduction_energy_rate_export",
            "dict",
            cond_rate_obj);
      }
      const py::dict cond_rate =
          py::reinterpret_borrow<py::dict>(cond_rate_obj);
      enforce_known_keys(
          cond_rate,
          "Numerics.diagnostics.conduction_energy_rate_export",
          {"enabled"});
      if (has_key(cond_rate, "enabled")) {
        numerics.diagnostics.conduction_energy_rate_export.enabled =
            strict_bool(
                cond_rate["enabled"],
                "Numerics.diagnostics.conduction_energy_rate_export.enabled");
      }
    }
    if (has_key(diagnostics, "mesh_quality_min")) {
      const py::handle mesh_quality_obj = diagnostics["mesh_quality_min"];
      if (!py::isinstance<py::dict>(mesh_quality_obj)) {
        throw_value_type_error("Numerics.diagnostics.mesh_quality_min",
                               "dict",
                               mesh_quality_obj);
      }
      const py::dict mesh_quality =
          py::reinterpret_borrow<py::dict>(mesh_quality_obj);
      enforce_known_keys(mesh_quality,
                         "Numerics.diagnostics.mesh_quality_min",
                         {"enabled"});
      if (has_key(mesh_quality, "enabled")) {
                        numerics.diagnostics.mesh_quality_min.enabled =
            strict_bool(mesh_quality["enabled"],
                        "Numerics.diagnostics.mesh_quality_min.enabled");
      }
    }
    if (has_key(diagnostics, "shock_approach")) {
      const py::handle shock_approach_obj = diagnostics["shock_approach"];
      if (!py::isinstance<py::dict>(shock_approach_obj)) {
        throw_value_type_error("Numerics.diagnostics.shock_approach",
                               "dict",
                               shock_approach_obj);
      }
      const py::dict shock_approach =
          py::reinterpret_borrow<py::dict>(shock_approach_obj);
      enforce_known_keys(shock_approach,
                         "Numerics.diagnostics.shock_approach",
                         {"enabled", "every", "target_radius_cm", "bins", "h_cell_cm",
                          "sectors", "modal_l_max", "sector_confidence_nu",
                          "sector_guard_crossings"});
      auto& cfg_shock_approach = numerics.diagnostics.shock_approach;
      if (has_key(shock_approach, "enabled")) {
        cfg_shock_approach.enabled = strict_bool(
            shock_approach["enabled"],
            "Numerics.diagnostics.shock_approach.enabled");
      }
      if (has_key(shock_approach, "every")) {
        cfg_shock_approach.every = strict_int32(
            shock_approach["every"],
            "Numerics.diagnostics.shock_approach.every");
      }
      if (has_key(shock_approach, "target_radius_cm")) {
        cfg_shock_approach.target_radius_cm = numeric_as_double(
            shock_approach["target_radius_cm"],
            "Numerics.diagnostics.shock_approach.target_radius_cm");
      }
      if (has_key(shock_approach, "bins")) {
        cfg_shock_approach.bins = strict_int32(
            shock_approach["bins"],
            "Numerics.diagnostics.shock_approach.bins");
      }
      if (has_key(shock_approach, "h_cell_cm")) {
        cfg_shock_approach.h_cell_cm = numeric_as_double(
            shock_approach["h_cell_cm"],
            "Numerics.diagnostics.shock_approach.h_cell_cm");
      }
      if (has_key(shock_approach, "sectors")) {
        cfg_shock_approach.sectors = strict_int32(
            shock_approach["sectors"],
            "Numerics.diagnostics.shock_approach.sectors");
      }
      if (has_key(shock_approach, "modal_l_max")) {
        cfg_shock_approach.modal_l_max = strict_int32(
            shock_approach["modal_l_max"],
            "Numerics.diagnostics.shock_approach.modal_l_max");
      }
      if (has_key(shock_approach, "sector_confidence_nu")) {
        cfg_shock_approach.sector_confidence_nu = numeric_as_double(
            shock_approach["sector_confidence_nu"],
            "Numerics.diagnostics.shock_approach.sector_confidence_nu");
      }
      if (has_key(shock_approach, "sector_guard_crossings")) {
        cfg_shock_approach.sector_guard_crossings = numeric_as_double(
            shock_approach["sector_guard_crossings"],
            "Numerics.diagnostics.shock_approach.sector_guard_crossings");
      }
      ensure_int_ge(cfg_shock_approach.every,
                    1,
                    "Numerics.diagnostics.shock_approach.every");
      ensure_int_ge(cfg_shock_approach.bins,
                    16,
                    "Numerics.diagnostics.shock_approach.bins");
      ensure_non_negative(
          cfg_shock_approach.h_cell_cm,
          "Numerics.diagnostics.shock_approach.h_cell_cm");
      ensure_int_ge(cfg_shock_approach.sectors,
                    0,
                    "Numerics.diagnostics.shock_approach.sectors");
      if (cfg_shock_approach.sectors > 0 &&
          (cfg_shock_approach.sectors < 4 ||
           cfg_shock_approach.sectors % 2 != 0)) {
        throw ValueError(
            "Numerics.diagnostics.shock_approach.sectors must be even and >= 4 when > 0");
      }
      if (cfg_shock_approach.modal_l_max < 0 ||
          cfg_shock_approach.modal_l_max > 4) {
        throw ValueError(
            "Numerics.diagnostics.shock_approach.modal_l_max must be in [0, 4]");
      }
      ensure_positive(
          cfg_shock_approach.sector_confidence_nu,
          "Numerics.diagnostics.shock_approach.sector_confidence_nu");
      ensure_non_negative(
          cfg_shock_approach.sector_guard_crossings,
          "Numerics.diagnostics.shock_approach.sector_guard_crossings");
      if (cfg_shock_approach.enabled &&
          !(cfg_shock_approach.target_radius_cm > 0.0)) {
        throw ConfigError(
            "Numerics.diagnostics.shock_approach.target_radius_cm must be > 0 when enabled");
      }
    }
    if (has_key(diagnostics, "ale_velcoherence")) {
      const py::handle vel_obj = diagnostics["ale_velcoherence"];
      if (!py::isinstance<py::dict>(vel_obj)) {
        throw_value_type_error("Numerics.diagnostics.ale_velcoherence",
                               "dict",
                               vel_obj);
      }
      const py::dict vel = py::reinterpret_borrow<py::dict>(vel_obj);
      enforce_known_keys(vel,
                         "Numerics.diagnostics.ale_velcoherence",
                         {"enabled", "every_n_steps"});
      auto& cfg_vel = numerics.diagnostics.ale_velcoherence;
      if (has_key(vel, "enabled")) {
        cfg_vel.enabled = strict_bool(
            vel["enabled"], "Numerics.diagnostics.ale_velcoherence.enabled");
      }
      if (has_key(vel, "every_n_steps")) {
        cfg_vel.every_n_steps = strict_int32(
            vel["every_n_steps"],
            "Numerics.diagnostics.ale_velcoherence.every_n_steps");
      }
    }
    if (has_key(diagnostics, "production_audit")) {
      const py::handle audit_obj = diagnostics["production_audit"];
      if (!py::isinstance<py::dict>(audit_obj)) {
        throw_value_type_error("Numerics.diagnostics.production_audit",
                               "dict",
                               audit_obj);
      }
      const py::dict audit = py::reinterpret_borrow<py::dict>(audit_obj);
      enforce_known_keys(audit,
                         "Numerics.diagnostics.production_audit",
                         {"enabled",
                          "tier",
                          "audit_json_path",
                          "escape_valve_budget",
                          "region_of_interest",
                          "gcl",
                          "positivity"});
      auto& cfg_audit = numerics.diagnostics.production_audit;
      if (has_key(audit, "enabled")) {
        cfg_audit.enabled = strict_bool(
            audit["enabled"],
            "Numerics.diagnostics.production_audit.enabled");
      }
      if (has_key(audit, "tier")) {
        cfg_audit.tier = strict_string(
            audit["tier"],
            "Numerics.diagnostics.production_audit.tier");
        if (!is_production_audit_tier(cfg_audit.tier)) {
          throw ValueError(
              "Numerics.diagnostics.production_audit.tier must be one of "
              "{\"A\", \"B\", \"none\"}");
        }
      }
      if (has_key(audit, "audit_json_path")) {
        cfg_audit.audit_json_path = strict_string(
            audit["audit_json_path"],
            "Numerics.diagnostics.production_audit.audit_json_path");
      }
      if (has_key(audit, "escape_valve_budget")) {
        const py::handle budget_obj = audit["escape_valve_budget"];
        if (!py::isinstance<py::dict>(budget_obj)) {
          throw_value_type_error(
              "Numerics.diagnostics.production_audit.escape_valve_budget",
              "dict",
              budget_obj);
        }
        const py::dict budget = py::reinterpret_borrow<py::dict>(budget_obj);
        enforce_known_keys(
            budget,
            "Numerics.diagnostics.production_audit.escape_valve_budget",
            {"mass_max", "energy_max"});
        if (has_key(budget, "mass_max")) {
          cfg_audit.escape_valve_budget.mass_max = numeric_as_double(
              budget["mass_max"],
              "Numerics.diagnostics.production_audit."
              "escape_valve_budget.mass_max");
          ensure_non_negative(
              cfg_audit.escape_valve_budget.mass_max,
              "Numerics.diagnostics.production_audit."
              "escape_valve_budget.mass_max");
        }
        if (has_key(budget, "energy_max")) {
          cfg_audit.escape_valve_budget.energy_max = numeric_as_double(
              budget["energy_max"],
              "Numerics.diagnostics.production_audit."
              "escape_valve_budget.energy_max");
          ensure_non_negative(
              cfg_audit.escape_valve_budget.energy_max,
              "Numerics.diagnostics.production_audit."
              "escape_valve_budget.energy_max");
        }
      }
      if (has_key(audit, "region_of_interest")) {
        parse_production_audit_regions(audit["region_of_interest"], cfg_audit);
      }
      if (has_key(audit, "gcl")) {
        const py::handle gcl_obj = audit["gcl"];
        if (!py::isinstance<py::dict>(gcl_obj)) {
          throw_value_type_error(
              "Numerics.diagnostics.production_audit.gcl",
              "dict",
              gcl_obj);
        }
        const py::dict gcl = py::reinterpret_borrow<py::dict>(gcl_obj);
        enforce_known_keys(gcl,
                           "Numerics.diagnostics.production_audit.gcl",
                           {"enabled"});
        if (has_key(gcl, "enabled")) {
          cfg_audit.gcl.enabled = strict_bool(
              gcl["enabled"],
              "Numerics.diagnostics.production_audit.gcl.enabled");
        }
      }
      if (has_key(audit, "positivity")) {
        const py::handle positivity_obj = audit["positivity"];
        if (!py::isinstance<py::dict>(positivity_obj)) {
          throw_value_type_error(
              "Numerics.diagnostics.production_audit.positivity",
              "dict",
              positivity_obj);
        }
        const py::dict positivity =
            py::reinterpret_borrow<py::dict>(positivity_obj);
        enforce_known_keys(
            positivity,
            "Numerics.diagnostics.production_audit.positivity",
            {"enabled", "fatal_on_neg"});
        if (has_key(positivity, "enabled")) {
          cfg_audit.positivity.enabled = strict_bool(
              positivity["enabled"],
              "Numerics.diagnostics.production_audit.positivity.enabled");
        }
        if (has_key(positivity, "fatal_on_neg")) {
          cfg_audit.positivity.fatal_on_neg = strict_bool(
              positivity["fatal_on_neg"],
              "Numerics.diagnostics.production_audit.positivity.fatal_on_neg");
        }
      }
    }
    if (has_key(diagnostics, "mesh_degeneracy_forensics")) {
      const py::handle forensics_obj = diagnostics["mesh_degeneracy_forensics"];
      if (!py::isinstance<py::dict>(forensics_obj)) {
        throw_value_type_error("Numerics.diagnostics.mesh_degeneracy_forensics",
                               "dict",
                               forensics_obj);
      }
      const py::dict forensics = py::reinterpret_borrow<py::dict>(forensics_obj);
      enforce_known_keys(forensics,
                         "Numerics.diagnostics.mesh_degeneracy_forensics",
                         {"enabled",
                          "corner_j_source_budget_enabled",
                          "corner_j_source_budget_include_1_ring",
                          "velocity_history_enabled",
                          "velocity_history_target_cell_c",
                          "velocity_history_sample_every_n_steps",
                          "velocity_history_include_1_ring",
                          "velocity_history_max_records",
                          "same_cell_count",
                          "sigma_threshold",
                          "max_dumps_per_run",
                          "output_dir"});
      auto& cfg_forensics = numerics.diagnostics.mesh_degeneracy_forensics;
      if (has_key(forensics, "enabled")) {
        cfg_forensics.enabled = strict_bool(
            forensics["enabled"],
            "Numerics.diagnostics.mesh_degeneracy_forensics.enabled");
      }
      if (has_key(forensics, "corner_j_source_budget_enabled")) {
        cfg_forensics.corner_j_source_budget_enabled = strict_bool(
            forensics["corner_j_source_budget_enabled"],
            "Numerics.diagnostics.mesh_degeneracy_forensics."
            "corner_j_source_budget_enabled");
      }
      if (has_key(forensics, "corner_j_source_budget_include_1_ring")) {
        cfg_forensics.corner_j_source_budget_include_1_ring = strict_bool(
            forensics["corner_j_source_budget_include_1_ring"],
            "Numerics.diagnostics.mesh_degeneracy_forensics."
            "corner_j_source_budget_include_1_ring");
      }
      if (has_key(forensics, "velocity_history_enabled")) {
        cfg_forensics.velocity_history_enabled = strict_bool(
            forensics["velocity_history_enabled"],
            "Numerics.diagnostics.mesh_degeneracy_forensics."
            "velocity_history_enabled");
      }
      if (has_key(forensics, "velocity_history_target_cell_c")) {
        cfg_forensics.velocity_history_target_cell_c = strict_int32(
            forensics["velocity_history_target_cell_c"],
            "Numerics.diagnostics.mesh_degeneracy_forensics."
            "velocity_history_target_cell_c");
        if (cfg_forensics.velocity_history_target_cell_c < -1) {
          throw ValueError(
              "Numerics.diagnostics.mesh_degeneracy_forensics."
              "velocity_history_target_cell_c must be >= -1");
        }
      }
      if (has_key(forensics, "velocity_history_sample_every_n_steps")) {
        cfg_forensics.velocity_history_sample_every_n_steps = strict_int32(
            forensics["velocity_history_sample_every_n_steps"],
            "Numerics.diagnostics.mesh_degeneracy_forensics."
            "velocity_history_sample_every_n_steps");
        if (cfg_forensics.velocity_history_sample_every_n_steps < 1) {
          throw ValueError(
              "Numerics.diagnostics.mesh_degeneracy_forensics."
              "velocity_history_sample_every_n_steps must be >= 1");
        }
      }
      if (has_key(forensics, "velocity_history_include_1_ring")) {
        cfg_forensics.velocity_history_include_1_ring = strict_bool(
            forensics["velocity_history_include_1_ring"],
            "Numerics.diagnostics.mesh_degeneracy_forensics."
            "velocity_history_include_1_ring");
      }
      if (has_key(forensics, "velocity_history_max_records")) {
        cfg_forensics.velocity_history_max_records = strict_int32(
            forensics["velocity_history_max_records"],
            "Numerics.diagnostics.mesh_degeneracy_forensics."
            "velocity_history_max_records");
        if (cfg_forensics.velocity_history_max_records < 0) {
          throw ValueError(
              "Numerics.diagnostics.mesh_degeneracy_forensics."
              "velocity_history_max_records must be >= 0");
        }
      }
      if (has_key(forensics, "same_cell_count")) {
        cfg_forensics.same_cell_count = strict_int32(
            forensics["same_cell_count"],
            "Numerics.diagnostics.mesh_degeneracy_forensics.same_cell_count");
        if (cfg_forensics.same_cell_count < 1) {
          throw ValueError(
              "Numerics.diagnostics.mesh_degeneracy_forensics.same_cell_count "
              "must be >= 1");
        }
      }
      if (has_key(forensics, "sigma_threshold")) {
        cfg_forensics.sigma_threshold = numeric_as_double(
            forensics["sigma_threshold"],
            "Numerics.diagnostics.mesh_degeneracy_forensics.sigma_threshold");
        if (!(cfg_forensics.sigma_threshold > 0.0 &&
              cfg_forensics.sigma_threshold <= 1.0)) {
          throw ValueError(
              "Numerics.diagnostics.mesh_degeneracy_forensics.sigma_threshold "
              "must be in (0, 1]");
        }
      }
      if (has_key(forensics, "max_dumps_per_run")) {
        cfg_forensics.max_dumps_per_run = strict_int32(
            forensics["max_dumps_per_run"],
            "Numerics.diagnostics.mesh_degeneracy_forensics.max_dumps_per_run");
        if (cfg_forensics.max_dumps_per_run < 0) {
          throw ValueError(
              "Numerics.diagnostics.mesh_degeneracy_forensics.max_dumps_per_run "
              "must be >= 0");
        }
      }
      if (has_key(forensics, "output_dir")) {
        cfg_forensics.output_dir = strict_string(
            forensics["output_dir"],
            "Numerics.diagnostics.mesh_degeneracy_forensics.output_dir");
      }
    }
  }

  if (has_key(kwargs, "profile")) {
    const py::handle profile_obj = kwargs["profile"];
    if (!py::isinstance<py::dict>(profile_obj)) {
      throw_value_type_error("Numerics.profile", "dict", profile_obj);
    }
    const py::dict profile = py::reinterpret_borrow<py::dict>(profile_obj);
    enforce_known_keys(profile,
                       "Numerics.profile",
                       {"icf_standard_ale", "legacy_regression"});
    if (has_key(profile, "icf_standard_ale")) {
      const py::handle icf_obj = profile["icf_standard_ale"];
      if (!py::isinstance<py::dict>(icf_obj)) {
        throw_value_type_error("Numerics.profile.icf_standard_ale",
                               "dict",
                               icf_obj);
      }
      const py::dict icf = py::reinterpret_borrow<py::dict>(icf_obj);
      enforce_known_keys(icf,
                         "Numerics.profile.icf_standard_ale",
                         {"enabled",
                          "enforce",
                          "claim_level",
                          "allowed_when_enabled",
                          "forbidden_when_enabled",
                          "escape_valves"});
      auto& cfg = numerics.profile.icf_standard_ale;
      const auto non_empty_string =
          [](const py::handle value, const std::string& path) {
            std::string out = strict_string(value, path);
            if (out.empty()) {
              throw ValueError(path + " must not be empty");
            }
            return out;
          };
      const auto string_vector =
          [&](const py::handle value, const std::string& path) {
            if (!py::isinstance<py::sequence>(value) ||
                py::isinstance<py::str>(value)) {
              throw_value_type_error(path, "list[str]", value);
            }
            std::vector<std::string> out;
            const auto seq = py::reinterpret_borrow<py::sequence>(value);
            out.reserve(seq.size());
            for (std::size_t i = 0; i < seq.size(); ++i) {
              const std::string item_path =
                  path + "[" + std::to_string(i) + "]";
              out.push_back(non_empty_string(seq[i], item_path));
            }
            return out;
          };
      const auto bool_vector =
          [&](const py::handle value, const std::string& path) {
            if (!py::isinstance<py::sequence>(value) ||
                py::isinstance<py::str>(value)) {
              throw_value_type_error(path, "list[bool]", value);
            }
            std::vector<bool> out;
            const auto seq = py::reinterpret_borrow<py::sequence>(value);
            out.reserve(seq.size());
            for (std::size_t i = 0; i < seq.size(); ++i) {
              const std::string item_path =
                  path + "[" + std::to_string(i) + "]";
              out.push_back(strict_bool(seq[i], item_path));
            }
            return out;
          };
      if (has_key(icf, "enabled")) {
        cfg.enabled = strict_bool(icf["enabled"],
                                  "Numerics.profile.icf_standard_ale.enabled");
      }
      if (has_key(icf, "enforce")) {
        cfg.enforce = strict_bool(icf["enforce"],
                                  "Numerics.profile.icf_standard_ale.enforce");
      }
      if (has_key(icf, "claim_level")) {
        cfg.claim_level = strict_string(
            icf["claim_level"],
            "Numerics.profile.icf_standard_ale.claim_level");
        if (!tenryu::core::is_icf_standard_ale_claim_level(cfg.claim_level)) {
          throw ConfigError(
              "Numerics.profile.icf_standard_ale.claim_level must be one of "
              "{\"characterization\", \"pre_plic_smoke\", "
              "\"production_comparable\"}");
        }
      }
      if (has_key(icf, "allowed_when_enabled")) {
        const py::handle allowed_obj = icf["allowed_when_enabled"];
        if (!py::isinstance<py::dict>(allowed_obj)) {
          throw_value_type_error(
              "Numerics.profile.icf_standard_ale.allowed_when_enabled",
              "dict",
              allowed_obj);
        }
        const py::dict allowed = py::reinterpret_borrow<py::dict>(allowed_obj);
        enforce_known_keys(
            allowed,
            "Numerics.profile.icf_standard_ale.allowed_when_enabled",
            {"ale_enabled_required_value",
             "ale_axis_repair_mode_required_value",
             "ale_remap_scheme_allowed_values",
             "ale_donor_sign_fixed",
             "hydro_driver_full_step_retry_enabled_required_value"});
        auto& allowed_cfg = cfg.allowed_when_enabled;
        if (has_key(allowed, "ale_enabled_required_value")) {
          allowed_cfg.ale_enabled_required_value = strict_bool(
              allowed["ale_enabled_required_value"],
              "Numerics.profile.icf_standard_ale.allowed_when_enabled."
              "ale_enabled_required_value");
        }
        if (has_key(allowed, "ale_axis_repair_mode_required_value")) {
          allowed_cfg.ale_axis_repair_mode_required_value = non_empty_string(
              allowed["ale_axis_repair_mode_required_value"],
              "Numerics.profile.icf_standard_ale.allowed_when_enabled."
              "ale_axis_repair_mode_required_value");
        }
        if (has_key(allowed, "ale_remap_scheme_allowed_values")) {
          allowed_cfg.ale_remap_scheme_allowed_values = string_vector(
              allowed["ale_remap_scheme_allowed_values"],
              "Numerics.profile.icf_standard_ale.allowed_when_enabled."
              "ale_remap_scheme_allowed_values");
        }
        if (has_key(allowed, "ale_donor_sign_fixed")) {
          allowed_cfg.ale_donor_sign_fixed_allowed_values = bool_vector(
              allowed["ale_donor_sign_fixed"],
              "Numerics.profile.icf_standard_ale.allowed_when_enabled."
              "ale_donor_sign_fixed");
        }
        if (has_key(allowed, "hydro_driver_full_step_retry_enabled_required_value")) {
          allowed_cfg.hydro_driver_full_step_retry_enabled_required_value =
              strict_bool(
                  allowed["hydro_driver_full_step_retry_enabled_required_value"],
                  "Numerics.profile.icf_standard_ale.allowed_when_enabled."
                  "hydro_driver_full_step_retry_enabled_required_value");
        }
      }
      if (has_key(icf, "forbidden_when_enabled")) {
        const py::handle forbidden_obj = icf["forbidden_when_enabled"];
        if (!py::isinstance<py::dict>(forbidden_obj)) {
          throw_value_type_error(
              "Numerics.profile.icf_standard_ale.forbidden_when_enabled",
              "dict",
              forbidden_obj);
        }
        const py::dict forbidden =
            py::reinterpret_borrow<py::dict>(forbidden_obj);
        enforce_known_keys(
            forbidden,
            "Numerics.profile.icf_standard_ale.forbidden_when_enabled",
            {"hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value",
             "ale_local_boundary_repair_enabled_forbidden_value",
             "ale_multi_node_boundary_repair_enabled_forbidden_value",
             "ale_multi_node_interior_repair_enabled_forbidden_value",
             "ale_axis_variational_projection_enabled_forbidden_value",
             "ale_emergency_cell_deactivation_enabled_forbidden_value",
             "hydro_driver_retry_active_mesh_repair_enabled_forbidden_value"});
        auto& forbidden_cfg = cfg.forbidden_when_enabled;
        if (has_key(forbidden,
                    "hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value")) {
          forbidden_cfg
              .hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value =
              strict_bool(
                  forbidden[
                      "hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value"],
                  "Numerics.profile.icf_standard_ale.forbidden_when_enabled."
                  "hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value");
        }
        if (has_key(forbidden, "ale_local_boundary_repair_enabled_forbidden_value")) {
          forbidden_cfg.ale_local_boundary_repair_enabled_forbidden_value =
              strict_bool(
                  forbidden["ale_local_boundary_repair_enabled_forbidden_value"],
                  "Numerics.profile.icf_standard_ale.forbidden_when_enabled."
                  "ale_local_boundary_repair_enabled_forbidden_value");
        }
        if (has_key(forbidden,
                    "ale_multi_node_boundary_repair_enabled_forbidden_value")) {
          forbidden_cfg.ale_multi_node_boundary_repair_enabled_forbidden_value =
              strict_bool(
                  forbidden[
                      "ale_multi_node_boundary_repair_enabled_forbidden_value"],
                  "Numerics.profile.icf_standard_ale.forbidden_when_enabled."
                  "ale_multi_node_boundary_repair_enabled_forbidden_value");
        }
        if (has_key(forbidden,
                    "ale_multi_node_interior_repair_enabled_forbidden_value")) {
          forbidden_cfg.ale_multi_node_interior_repair_enabled_forbidden_value =
              strict_bool(
                  forbidden[
                      "ale_multi_node_interior_repair_enabled_forbidden_value"],
                  "Numerics.profile.icf_standard_ale.forbidden_when_enabled."
                  "ale_multi_node_interior_repair_enabled_forbidden_value");
        }
        if (has_key(forbidden,
                    "ale_axis_variational_projection_enabled_forbidden_value")) {
          forbidden_cfg.ale_axis_variational_projection_enabled_forbidden_value =
              strict_bool(
                  forbidden[
                      "ale_axis_variational_projection_enabled_forbidden_value"],
                  "Numerics.profile.icf_standard_ale.forbidden_when_enabled."
                  "ale_axis_variational_projection_enabled_forbidden_value");
        }
        if (has_key(forbidden,
                    "ale_emergency_cell_deactivation_enabled_forbidden_value")) {
          forbidden_cfg.ale_emergency_cell_deactivation_enabled_forbidden_value =
              strict_bool(
                  forbidden[
                      "ale_emergency_cell_deactivation_enabled_forbidden_value"],
                  "Numerics.profile.icf_standard_ale.forbidden_when_enabled."
                  "ale_emergency_cell_deactivation_enabled_forbidden_value");
        }
        if (has_key(
                forbidden,
                "hydro_driver_retry_active_mesh_repair_enabled_forbidden_value")) {
          forbidden_cfg
              .hydro_driver_retry_active_mesh_repair_enabled_forbidden_value =
              strict_bool(
                  forbidden[
                      "hydro_driver_retry_active_mesh_repair_enabled_forbidden_value"],
                  "Numerics.profile.icf_standard_ale.forbidden_when_enabled."
                  "hydro_driver_retry_active_mesh_repair_enabled_forbidden_value");
        }
      }
      if (has_key(icf, "escape_valves")) {
        const py::handle escape_obj = icf["escape_valves"];
        if (!py::isinstance<py::dict>(escape_obj)) {
          throw_value_type_error(
              "Numerics.profile.icf_standard_ale.escape_valves",
              "dict",
              escape_obj);
        }
        const py::dict escape = py::reinterpret_borrow<py::dict>(escape_obj);
        enforce_known_keys(
            escape,
            "Numerics.profile.icf_standard_ale.escape_valves",
            {"allow_nonstandard_mesh_rescue",
             "require_deck_reason",
             "mark_run_nonstandard"});
        auto& escape_cfg = cfg.escape_valves;
        if (has_key(escape, "allow_nonstandard_mesh_rescue")) {
          escape_cfg.allow_nonstandard_mesh_rescue = strict_bool(
              escape["allow_nonstandard_mesh_rescue"],
              "Numerics.profile.icf_standard_ale.escape_valves."
              "allow_nonstandard_mesh_rescue");
        }
        if (has_key(escape, "require_deck_reason")) {
          escape_cfg.require_deck_reason = strict_bool(
              escape["require_deck_reason"],
              "Numerics.profile.icf_standard_ale.escape_valves."
              "require_deck_reason");
        }
        if (has_key(escape, "mark_run_nonstandard")) {
          escape_cfg.mark_run_nonstandard = strict_bool(
              escape["mark_run_nonstandard"],
              "Numerics.profile.icf_standard_ale.escape_valves."
              "mark_run_nonstandard");
        }
      }
    }
    if (has_key(profile, "legacy_regression")) {
      const py::handle legacy_obj = profile["legacy_regression"];
      if (!py::isinstance<py::dict>(legacy_obj)) {
        throw_value_type_error("Numerics.profile.legacy_regression",
                               "dict",
                               legacy_obj);
      }
      const py::dict legacy =
          py::reinterpret_borrow<py::dict>(legacy_obj);
      enforce_known_keys(legacy,
                         "Numerics.profile.legacy_regression",
                         {"enabled", "revision"});
      auto& cfg = numerics.profile.legacy_regression;
      if (has_key(legacy, "enabled")) {
        cfg.enabled = strict_bool(
            legacy["enabled"],
            "Numerics.profile.legacy_regression.enabled");
      }
      if (has_key(legacy, "revision")) {
        cfg.revision = strict_string(
            legacy["revision"],
            "Numerics.profile.legacy_regression.revision");
      }
    }
  }

  if (has_key(kwargs, "dt")) {
    const py::handle dt_obj = kwargs["dt"];
    if (!py::isinstance<py::dict>(dt_obj)) {
      throw_value_type_error("Numerics.dt", "dict", dt_obj);
    }
    const py::dict dt = py::reinterpret_borrow<py::dict>(dt_obj);
    enforce_known_keys(dt, "Numerics.dt",
                       {"initial_s", "cfl_hydro", "cfl_length_2d", "cfl_cond",
                        "edge_accel_displacement_cfl_enabled",
                        "f_min_fleck",
                        "growth_factor", "max_s", "min_s",
                        "min_consecutive_steps",
                        "floor_stall_max_consecutive_steps"});
    if (has_key(dt, "initial_s")) {
      if (dt["initial_s"].is_none()) {
        numerics.dt.initial_s = -1.0;
      } else {
        numerics.dt.initial_s =
            numeric_as_double(dt["initial_s"], "Numerics.dt.initial_s");
      }
    }
    if (has_key(dt, "cfl_hydro")) {
      numerics.dt.cfl_hydro = numeric_as_double(dt["cfl_hydro"], "Numerics.dt.cfl_hydro");
    }
    if (has_key(dt, "cfl_length_2d")) {
      numerics.dt.cfl_length_2d =
          strict_string(dt["cfl_length_2d"], "Numerics.dt.cfl_length_2d");
      if (numerics.dt.cfl_length_2d != "sqrt_area" &&
          numerics.dt.cfl_length_2d != "min_altitude") {
        throw ValueError(
            "Numerics.dt.cfl_length_2d must be one of "
            "{\"sqrt_area\", \"min_altitude\"}, got " +
            numerics.dt.cfl_length_2d);
      }
    }
    if (has_key(dt, "edge_accel_displacement_cfl_enabled")) {
      numerics.dt.edge_accel_displacement_cfl_enabled = strict_bool(
          dt["edge_accel_displacement_cfl_enabled"],
          "Numerics.dt.edge_accel_displacement_cfl_enabled");
    }
    if (has_key(dt, "cfl_cond")) {
      numerics.dt.cfl_cond = numeric_as_double(dt["cfl_cond"], "Numerics.dt.cfl_cond");
    }
    if (has_key(dt, "f_min_fleck")) {
      numerics.dt.f_min_fleck =
          numeric_as_double(dt["f_min_fleck"], "Numerics.dt.f_min_fleck");
    }
    if (has_key(dt, "growth_factor")) {
      numerics.dt.growth_factor =
          numeric_as_double(dt["growth_factor"], "Numerics.dt.growth_factor");
    }
    if (has_key(dt, "max_s")) {
      numerics.dt.max_s = numeric_as_double(dt["max_s"], "Numerics.dt.max_s");
    }
    if (has_key(dt, "min_s")) {
      numerics.dt.min_s = numeric_as_double(dt["min_s"], "Numerics.dt.min_s");
    }
    if (has_key(dt, "min_consecutive_steps")) {
      numerics.dt.min_consecutive_steps = strict_int32(
          dt["min_consecutive_steps"], "Numerics.dt.min_consecutive_steps");
    }
    if (has_key(dt, "floor_stall_max_consecutive_steps")) {
      numerics.dt.floor_stall_max_consecutive_steps = strict_int32(
          dt["floor_stall_max_consecutive_steps"],
          "Numerics.dt.floor_stall_max_consecutive_steps");
      if (numerics.dt.floor_stall_max_consecutive_steps < 0) {
        throw ValueError(
            "Numerics.dt.floor_stall_max_consecutive_steps must be >= 0");
      }
    }
  }

  if (has_key(kwargs, "hydro")) {
    const py::handle hydro_obj = kwargs["hydro"];
    if (!py::isinstance<py::dict>(hydro_obj)) {
      throw_value_type_error("Numerics.hydro", "dict", hydro_obj);
    }
    const py::dict hydro = py::reinterpret_borrow<py::dict>(hydro_obj);
    enforce_known_keys(hydro, "Numerics.hydro",
                       {"enabled", "compatible_energy", "T_start_eV", "boundary", "boundary_1d", "boundary_2d",
                       "av_type", "av_model", "rz_momentum_scheme", "corner_mass_convention",
                       "time_integration", "total_energy_identity_check",
                       "rz_momentum_scheme",
                       "axis_node_mass_convention",
                       "av_C1", "av_C2", "av_linear", "av_quadratic",
                       "csw98_degenerate_side_floor_rel",
                       "csw98_damper_impulse_beta",
                       "csw98_axisline_av_mode",
                       "csw98_axisline_d1prime_cfl_enabled",
                       "csw98_limiter_shock_floor_enabled",
                       "csw98_axisline_work_planar_enabled",
                       "tensor_av_C1", "tensor_av_C2",
                       "av_qcap_over_p", "av_qcap_center_band_only",
                       "av_cfl_coefficient",
                       "csw_C1", "csw_C2", "csw_limiter",
                       "csw_limiter_enabled",
                       "csw_axis_mirror_limiter",
                       "csw_rz_lift_enabled", "csw_rz_lift_guard_ratio",
                       "csw_pole_floor_enabled", "csw_pole_floor_sigma0",
                       "csw_pole_floor_theta0_rad", "csw_pole_floor_thetaf_rad",
                       "csw_pole_desens_enabled", "csw_pole_desens_alpha",
                       "csw_pole_desens_theta0_rad", "csw_pole_desens_thetaf_rad",
                       "csw_polar_slaving_enabled",
                       "csw_polar_slaving_min_columns",
                       "csw_polar_slaving_full_columns",
                       "csw_polar_slaving_outer_columns",
                       "csw_polar_slaving_chi_on",
                       "csw_polar_slaving_chi_full",
                       "csw_polar_slaving_strength",
                       "csw_polar_slaving_av_stiffness_cfl_enabled",
                       "csw_polar_slaving_av_stiffness_sigma",
                       "wake_heat_flux_enabled", "wake_heat_flux_CE",
                       "wake_heat_flux_theta_a_rad", "wake_heat_flux_theta_b_rad",
                       "wake_heat_flux_global_theta",
                       "csw_shock_limiter_floor", "csw_zero_uniform_compression",
                       "csw_diagnostics",
                       "av_limiter_J", "av_heat_C", "post_shock_heat",
                        "post_shock_heat_C", "post_shock_heat_decay",
                        "post_shock_velocity_damping_C", "bulk_viscosity_C",
                        "ion_art_heat_C",
                        "crossing_dt_safety", "time_integrator",
                        "bbs_axis_policy_enabled",
                        "subzonal_mass_enabled",
                        "subzonal_mass_lagrangian_invariant_enabled",
                        "anti_hourglass_kappa",
                        "subzonal_pressure_enabled",
                        "pentagon_affine_null_enabled",
                        "pentagon_affine_null_kappa",
                        "subzonal_dt_limiter_enabled",
                        "aw_compatible_force_work",
                        "subzonal_pressure_mode",
                        "subzonal_band_mode",
                        "subzonal_band_feather_layers",
                        "subzonal_merit_mode",
                        "subzonal_alpha1",
                        "subzonal_alpha2",
                        "subzonal_merit_power",
                        "subzonal_merit_constant",
                        "hourglass", "axis_projection", "adaptive_av",
                        "plasma_viscosity",
                        "av_eos_aware",
                        "av_eos_gamma1_ref", "av_eos_boost_max",
                        "odd_even_damping_C", "ee_odd_even_C",
                        "hk_velocity_damper_C", "hk_velocity_damper_tau_min",
                        "hk_velocity_damper_grad_Te_max",
                        "hk_velocity_damper_grad_rho_max",
                        "hk_velocity_damper_guard_cells", "av_heat_to",
                        "boundary_pressure", "pressure_drive_perturbation",
                        "rho_e_linear_grid", "eos_writeback", "eos_closure_mode",
                        "qei_evaluate_at_t_n", "qei_multiplier", "exact_override",
                        "total_energy_remap_2d_rz",
                        "work_split_audit_2d_rz",
                        "work_split_audit_cell_every_n_steps",
                        "work_split_audit_all_rows",
                        "hllc_z_flux_2d_rz",
                        "hllc_z_flux_audit_2d_rz",
                        "hllc_z_flux_hlle_fallback",
                        "hllc_z_flux_strict_quasi_1d",
                        "axis_motion_floor_fraction",
                        "axis_margin_dt_floor_fraction",
                        "volume_rate_cfl_enabled",
                        "volume_rate_cfl_threshold",
                        "tri_fan_center_cfl_enabled",
                        "tri_fan_center_cfl_safety",
                        "tri_fan_center_cfl_band_radial_index",
                        "corner_j_predict_cfl_enabled",
                        "corner_j_predict_cfl_safety",
                        "corner_j_predict_floor_frac",
                        "corner_j_predict_max_shrink",
                        "corner_j_predict_shell_rings",
                        "tri_fan_center_perturbation_diag_enabled",
                        "av_qcap_scope",
                        "center_cfl_scope",
                        "center_perturbation_diag_scope",
                        "center_perturbation_diag_radial_bins",
                        "rz_geometric_cfl_enabled",
                        "rz_geometric_cfl_etaV",
                        "rz_geometric_cfl_r_floor",
                        "rz_geometric_cfl_cumulative_protection_enabled",
                        "rz_geometric_cfl_v_initial_floor",
                        "rz_geometric_cfl_precise_u_half_enabled",
                        "trial_volume_cfl_enabled",
                        "trial_volume_cfl_floor_fraction",
                        "trial_volume_cfl_shrink_fraction",
                        "corner_jacobian_ale_trigger_enabled",
                        "corner_jacobian_floor_eps",
                        "corner_jacobian_ale_trigger_scale",
                        "in_hydro_corner_j_guard_enabled",
                        "in_hydro_gauss_j_guard_enabled",
                        "in_hydro_rz_volume_guard_enabled",
                        "in_hydro_gauss_j_floor_rel",
                        "in_hydro_rz_volume_floor_rel",
                        "mesh_quality_dt_cfl_enabled",
                        "mesh_quality_dt_safety_alpha",
                        "mesh_quality_dt_corner_j_enabled",
                        "mesh_quality_dt_gauss_j_enabled",
                        "mesh_quality_dt_rz_volume_enabled",
                        "mesh_quality_dt_axis_margin_additive",
                        "mesh_quality_dt_corner_j_floor_rel",
                        "mesh_quality_dt_gauss_j_floor_rel",
                        "mesh_quality_dt_rz_volume_floor_rel",
                        "ring7_quotient_enabled",
                        "regime_aware_corner_j_guard_enabled",
                        "axis_margin_guard_enabled",
                        "axis_margin_additive_in_action8_enabled",
                        "axis_guard_band_cells",
                        "driver_full_step_retry_enabled",
                        "driver_full_step_retry_max_attempts",
                        "dispatcher_state_sensitive_bypass_enabled",
                        "dispatcher_state_sensitive_repair_cap_per_step",
                        "strategy_first_retry_enabled",
                        "strategy_first_max_same_dt_attempts",
                        "driver_retry_active_mesh_repair_enabled",
                        "driver_retry_corner_balance_threshold",
                        "cascade_on_hydro_retry_enabled",
                        "driver_retry_use_suggested_dt_enabled",
                        "geometric_retry_stagnation",
                        "mesh_geometry_soft_fail_enabled"});
    if (has_key(hydro, "enabled")) {
      numerics.hydro.enabled = strict_bool(hydro["enabled"], "Numerics.hydro.enabled");
    }
    if (has_key(hydro, "compatible_energy")) {
      numerics.hydro.compatible_energy =
          strict_bool(hydro["compatible_energy"], "Numerics.hydro.compatible_energy");
    }
    if (has_key(hydro, "av_type")) {
      numerics.hydro.av_type_explicit = true;
      numerics.hydro.av_type =
          strict_string(hydro["av_type"], "Numerics.hydro.av_type");
      if (!is_hydro_av_type(numerics.hydro.av_type)) {
        throw ValueError(
            "Numerics.hydro.av_type must be one of {\"vnr\", \"riemann\", "
            "\"riemann_compatible\", \"csw\"}, got " +
            numerics.hydro.av_type);
      }
    }
    if (has_key(hydro, "av_model")) {
      const std::string av_model =
          strict_string(hydro["av_model"], "Numerics.hydro.av_model");
      if (!av_model_from_string(av_model, numerics.hydro.av_model)) {
        throw ValueError(
            "Numerics.hydro.av_model must be one of "
            "{\"scalar_vnr_legacy\", \"csw_edge\", \"csw_edge_csw98\", "
            "\"csw_edge_plus_tensor_limited\", \"mimetic_tensor_v1\"}, got " +
            av_model);
      }
    }
    if (has_key(hydro, "corner_mass_convention")) {
      const std::string corner_mass_convention =
          strict_string(hydro["corner_mass_convention"],
                        "Numerics.hydro.corner_mass_convention");
      if (!corner_mass_convention_from_string(
              corner_mass_convention,
              numerics.hydro.corner_mass_convention)) {
        throw ValueError(
            "Numerics.hydro.corner_mass_convention must be one of "
            "{\"bbsw_radial_v0\", \"kinematic_basis_rz_v1\"}, got " +
            corner_mass_convention);
      }
    }
    if (has_key(hydro, "time_integration")) {
      const std::string time_integration =
          strict_string(hydro["time_integration"],
                        "Numerics.hydro.time_integration");
      if (!hydro_time_integration_from_string(
              time_integration, numerics.hydro.time_integration)) {
        throw ValueError(
            "Numerics.hydro.time_integration must be one of "
            "{\"pc_v0\", \"midpoint_v1\"}, got " +
            time_integration);
      }
    }
    if (has_key(hydro, "total_energy_identity_check")) {
      numerics.hydro.total_energy_identity_check = strict_bool(
          hydro["total_energy_identity_check"],
          "Numerics.hydro.total_energy_identity_check");
    }
    if (has_key(hydro, "rz_momentum_scheme")) {
      numerics.hydro.rz_momentum_scheme = strict_string(
          hydro["rz_momentum_scheme"],
          "Numerics.hydro.rz_momentum_scheme");
      if (numerics.hydro.rz_momentum_scheme == "volume_weighted") {
        numerics.hydro.rz_momentum_scheme_id = 0;
      } else if (numerics.hydro.rz_momentum_scheme ==
                 "area_weighted_symmetric") {
        numerics.hydro.rz_momentum_scheme_id = 1;
      } else {
        throw ValueError(
            "Numerics.hydro.rz_momentum_scheme must be one of "
            "{\"volume_weighted\", \"area_weighted_symmetric\"}, got " +
            numerics.hydro.rz_momentum_scheme);
      }
    }
    if (has_key(hydro, "axis_node_mass_convention")) {
      numerics.hydro.axis_node_mass_convention = strict_string(
          hydro["axis_node_mass_convention"],
          "Numerics.hydro.axis_node_mass_convention");
      if (numerics.hydro.axis_node_mass_convention != "corner_subzonal" &&
          numerics.hydro.axis_node_mass_convention != "equal_split" &&
          numerics.hydro.axis_node_mass_convention != "equal_split_all") {
        throw ConfigError(
            "Numerics.hydro.axis_node_mass_convention must be \"corner_subzonal\", \"equal_split\", or \"equal_split_all\"");
      }
    }
    if (has_key(hydro, "rho_e_linear_grid")) {
      numerics.hydro.rho_e_linear_grid =
          strict_bool(hydro["rho_e_linear_grid"], "Numerics.hydro.rho_e_linear_grid");
    }
    if (has_key(hydro, "eos_writeback")) {
      numerics.hydro.eos_writeback =
          strict_bool(hydro["eos_writeback"], "Numerics.hydro.eos_writeback");
    }
    if (has_key(hydro, "eos_closure_mode")) {
      numerics.hydro.eos_closure_mode = strict_string(
          hydro["eos_closure_mode"], "Numerics.hydro.eos_closure_mode");
      if (numerics.hydro.eos_closure_mode != "legacy" &&
          numerics.hydro.eos_closure_mode != "energy_authoritative") {
        throw ConfigError(
            "Numerics.hydro.eos_closure_mode must be \"legacy\" or"
            " \"energy_authoritative\"");
      }
    }
    if (has_key(hydro, "qei_evaluate_at_t_n")) {
      numerics.hydro.qei_evaluate_at_t_n = strict_bool(
          hydro["qei_evaluate_at_t_n"],
          "Numerics.hydro.qei_evaluate_at_t_n");
    }
    if (has_key(hydro, "qei_multiplier")) {
      numerics.hydro.qei_multiplier =
          numeric_as_double(hydro["qei_multiplier"], "Numerics.hydro.qei_multiplier");
      if (!(std::isfinite(numerics.hydro.qei_multiplier) &&
            numerics.hydro.qei_multiplier > 0.0)) {
        throw ValueError("Numerics.hydro.qei_multiplier must be > 0");
      }
    }
    if (has_key(hydro, "exact_override")) {
      numerics.hydro.exact_override =
          strict_string(hydro["exact_override"], "Numerics.hydro.exact_override");
      if (!is_hydro_exact_override(numerics.hydro.exact_override)) {
        throw ValueError(
            "Numerics.hydro.exact_override must be one of {\"none\", \"pressure\", "
            "\"sound_speed\", \"temperature\", \"cv\", \"temp_reclosure\", "
            "\"pressure_and_cs\", \"p_t_cs\", \"no_writeback\"}, got " +
            numerics.hydro.exact_override);
      }
    }
    if (has_key(hydro, "total_energy_remap_2d_rz")) {
      numerics.hydro.total_energy_remap_2d_rz = strict_bool(
          hydro["total_energy_remap_2d_rz"],
          "Numerics.hydro.total_energy_remap_2d_rz");
    }
    if (has_key(hydro, "work_split_audit_2d_rz")) {
      numerics.hydro.work_split_audit_2d_rz = strict_bool(
          hydro["work_split_audit_2d_rz"],
          "Numerics.hydro.work_split_audit_2d_rz");
    }
    if (has_key(hydro, "work_split_audit_cell_every_n_steps")) {
      numerics.hydro.work_split_audit_cell_every_n_steps = strict_int32(
          hydro["work_split_audit_cell_every_n_steps"],
          "Numerics.hydro.work_split_audit_cell_every_n_steps");
      ensure_non_negative(numerics.hydro.work_split_audit_cell_every_n_steps,
                          "Numerics.hydro.work_split_audit_cell_every_n_steps");
    }
    if (has_key(hydro, "work_split_audit_all_rows")) {
      numerics.hydro.work_split_audit_all_rows = strict_bool(
          hydro["work_split_audit_all_rows"],
          "Numerics.hydro.work_split_audit_all_rows");
    }
    if (has_key(hydro, "hllc_z_flux_2d_rz")) {
      numerics.hydro.hllc_z_flux_2d_rz = strict_bool(
          hydro["hllc_z_flux_2d_rz"], "Numerics.hydro.hllc_z_flux_2d_rz");
    }
    if (has_key(hydro, "hllc_z_flux_audit_2d_rz")) {
      numerics.hydro.hllc_z_flux_audit_2d_rz = strict_bool(
          hydro["hllc_z_flux_audit_2d_rz"],
          "Numerics.hydro.hllc_z_flux_audit_2d_rz");
    }
    if (has_key(hydro, "hllc_z_flux_hlle_fallback")) {
      numerics.hydro.hllc_z_flux_hlle_fallback = strict_bool(
          hydro["hllc_z_flux_hlle_fallback"],
          "Numerics.hydro.hllc_z_flux_hlle_fallback");
    }
    if (has_key(hydro, "hllc_z_flux_strict_quasi_1d")) {
      numerics.hydro.hllc_z_flux_strict_quasi_1d = strict_bool(
          hydro["hllc_z_flux_strict_quasi_1d"],
          "Numerics.hydro.hllc_z_flux_strict_quasi_1d");
    }
    if (has_key(hydro, "bbs_axis_policy_enabled")) {
      numerics.hydro.bbs_axis_policy_enabled = strict_bool(
          hydro["bbs_axis_policy_enabled"],
          "Numerics.hydro.bbs_axis_policy_enabled");
    }
    if (has_key(hydro, "subzonal_mass_enabled")) {
      numerics.hydro.subzonal_mass_enabled = strict_bool(
          hydro["subzonal_mass_enabled"],
          "Numerics.hydro.subzonal_mass_enabled");
    }
    if (has_key(hydro, "subzonal_mass_lagrangian_invariant_enabled")) {
      numerics.hydro.subzonal_mass_lagrangian_invariant_enabled = strict_bool(
          hydro["subzonal_mass_lagrangian_invariant_enabled"],
          "Numerics.hydro.subzonal_mass_lagrangian_invariant_enabled");
    }
    if (has_key(hydro, "anti_hourglass_kappa")) {
      numerics.hydro.anti_hourglass_kappa = numeric_as_double(
          hydro["anti_hourglass_kappa"],
          "Numerics.hydro.anti_hourglass_kappa");
      ensure_positive(numerics.hydro.anti_hourglass_kappa,
                      "Numerics.hydro.anti_hourglass_kappa");
    }
    if (has_key(hydro, "subzonal_pressure_enabled")) {
      numerics.hydro.subzonal_pressure_enabled = strict_bool(
          hydro["subzonal_pressure_enabled"],
          "Numerics.hydro.subzonal_pressure_enabled");
    }
    if (has_key(hydro, "pentagon_affine_null_enabled")) {
      numerics.hydro.pentagon_affine_null_enabled = strict_bool(
          hydro["pentagon_affine_null_enabled"],
          "Numerics.hydro.pentagon_affine_null_enabled");
    }
    if (has_key(hydro, "pentagon_affine_null_kappa")) {
      numerics.hydro.pentagon_affine_null_kappa = numeric_as_double(
          hydro["pentagon_affine_null_kappa"],
          "Numerics.hydro.pentagon_affine_null_kappa");
      if (!(std::isfinite(numerics.hydro.pentagon_affine_null_kappa) &&
            numerics.hydro.pentagon_affine_null_kappa >= 0.0 &&
            numerics.hydro.pentagon_affine_null_kappa <= 1.0)) {
        throw ValueError(
            "Numerics.hydro.pentagon_affine_null_kappa must be finite and "
            "in [0, 1]");
      }
    }
    if (has_key(hydro, "subzonal_dt_limiter_enabled")) {
      numerics.hydro.subzonal_dt_limiter_enabled = strict_bool(
          hydro["subzonal_dt_limiter_enabled"],
          "Numerics.hydro.subzonal_dt_limiter_enabled");
    }
    if (has_key(hydro, "aw_compatible_force_work")) {
      numerics.hydro.aw_compatible_force_work = strict_bool(
          hydro["aw_compatible_force_work"],
          "Numerics.hydro.aw_compatible_force_work");
    }
    if (has_key(hydro, "subzonal_pressure_mode")) {
      numerics.hydro.subzonal_pressure_mode = strict_string(
          hydro["subzonal_pressure_mode"],
          "Numerics.hydro.subzonal_pressure_mode");
      if (numerics.hydro.subzonal_pressure_mode != "uniform_cell" &&
          numerics.hydro.subzonal_pressure_mode != "caramana_shashkov") {
        throw ValueError(
            "Numerics.hydro.subzonal_pressure_mode must be one of "
            "{\"uniform_cell\", \"caramana_shashkov\"}");
      }
      if (numerics.hydro.subzonal_pressure_mode == "caramana_shashkov") {
        throw ConfigError(
            "Numerics.hydro.subzonal_pressure_mode=\"caramana_shashkov\" "
            "is reserved for a future implementation");
      }
    }
    if (has_key(hydro, "subzonal_band_mode")) {
      numerics.hydro.subzonal_band_mode = strict_string(
          hydro["subzonal_band_mode"],
          "Numerics.hydro.subzonal_band_mode");
      if (numerics.hydro.subzonal_band_mode != "off" &&
          numerics.hydro.subzonal_band_mode != "bridge_feather") {
        throw ValueError(
            "Numerics.hydro.subzonal_band_mode must be one of "
            "{\"off\", \"bridge_feather\"}");
      }
    }
    if (has_key(hydro, "subzonal_band_feather_layers")) {
      numerics.hydro.subzonal_band_feather_layers = strict_int32(
          hydro["subzonal_band_feather_layers"],
          "Numerics.hydro.subzonal_band_feather_layers");
      if (numerics.hydro.subzonal_band_feather_layers < 1) {
        throw ValueError(
            "Numerics.hydro.subzonal_band_feather_layers must be >= 1");
      }
    }
    if (has_key(hydro, "subzonal_merit_mode")) {
      numerics.hydro.subzonal_merit_mode = strict_string(
          hydro["subzonal_merit_mode"], "Numerics.hydro.subzonal_merit_mode");
      if (!is_subzonal_merit_mode_config(numerics.hydro.subzonal_merit_mode)) {
        throw ValueError(
            "Numerics.hydro.subzonal_merit_mode must be one of "
            "{\"caramana_auto\", \"constant\", \"off\"}");
      }
    }
    if (has_key(hydro, "subzonal_alpha1")) {
      numerics.hydro.subzonal_alpha1 =
          numeric_as_double(hydro["subzonal_alpha1"],
                            "Numerics.hydro.subzonal_alpha1");
      ensure_positive(numerics.hydro.subzonal_alpha1,
                      "Numerics.hydro.subzonal_alpha1");
    }
    if (has_key(hydro, "subzonal_alpha2")) {
      numerics.hydro.subzonal_alpha2 =
          numeric_as_double(hydro["subzonal_alpha2"],
                            "Numerics.hydro.subzonal_alpha2");
      ensure_non_negative(numerics.hydro.subzonal_alpha2,
                          "Numerics.hydro.subzonal_alpha2");
    }
    if (has_key(hydro, "subzonal_merit_power")) {
      numerics.hydro.subzonal_merit_power =
          strict_int32(hydro["subzonal_merit_power"],
                       "Numerics.hydro.subzonal_merit_power");
      ensure_int_ge(numerics.hydro.subzonal_merit_power, 0,
                    "Numerics.hydro.subzonal_merit_power");
    }
    if (has_key(hydro, "subzonal_merit_constant")) {
      numerics.hydro.subzonal_merit_constant =
          numeric_as_double(hydro["subzonal_merit_constant"],
                            "Numerics.hydro.subzonal_merit_constant");
      ensure_non_negative(numerics.hydro.subzonal_merit_constant,
                          "Numerics.hydro.subzonal_merit_constant");
    }
    if (has_key(hydro, "hourglass")) {
      const py::handle hourglass_obj = hydro["hourglass"];
      if (!py::isinstance<py::dict>(hourglass_obj)) {
        throw_value_type_error("Numerics.hydro.hourglass", "dict", hourglass_obj);
      }
      const py::dict hourglass = py::reinterpret_borrow<py::dict>(hourglass_obj);
      enforce_known_keys(hourglass, "Numerics.hydro.hourglass",
                         {"enabled", "scale", "compatible_work_enabled",
                          "activation_corner_j_ratio_threshold",
                          "activation_hourglass_amplitude_threshold",
                          "subzonal_pressure_model",
                          "max_force_per_node_fraction"});
      auto& hg = numerics.hydro.hourglass;
      if (has_key(hourglass, "enabled")) {
        hg.enabled =
            strict_bool(hourglass["enabled"], "Numerics.hydro.hourglass.enabled");
      }
      if (has_key(hourglass, "scale")) {
        hg.scale = numeric_as_double(hourglass["scale"],
                                     "Numerics.hydro.hourglass.scale");
        ensure_positive(hg.scale, "Numerics.hydro.hourglass.scale");
      }
      if (has_key(hourglass, "compatible_work_enabled")) {
        hg.compatible_work_enabled = strict_bool(
            hourglass["compatible_work_enabled"],
            "Numerics.hydro.hourglass.compatible_work_enabled");
      }
      if (has_key(hourglass, "activation_corner_j_ratio_threshold")) {
        hg.activation_corner_j_ratio_threshold = numeric_as_double(
            hourglass["activation_corner_j_ratio_threshold"],
            "Numerics.hydro.hourglass.activation_corner_j_ratio_threshold");
        if (!(hg.activation_corner_j_ratio_threshold > 0.0 &&
              hg.activation_corner_j_ratio_threshold <= 1.0)) {
          throw ValueError(
              "Numerics.hydro.hourglass.activation_corner_j_ratio_threshold "
              "must be in (0, 1]");
        }
      }
      if (has_key(hourglass, "activation_hourglass_amplitude_threshold")) {
        hg.activation_hourglass_amplitude_threshold = numeric_as_double(
            hourglass["activation_hourglass_amplitude_threshold"],
            "Numerics.hydro.hourglass.activation_hourglass_amplitude_threshold");
        if (!(hg.activation_hourglass_amplitude_threshold > 0.0 &&
              hg.activation_hourglass_amplitude_threshold <= 1.0)) {
          throw ValueError(
              "Numerics.hydro.hourglass.activation_hourglass_amplitude_threshold "
              "must be in (0, 1]");
        }
      }
      if (has_key(hourglass, "subzonal_pressure_model")) {
        hg.subzonal_pressure_model = strict_string(
            hourglass["subzonal_pressure_model"],
            "Numerics.hydro.hourglass.subzonal_pressure_model");
        if (hg.subzonal_pressure_model != "linearized" &&
            hg.subzonal_pressure_model != "eos_lookup") {
          throw ValueError(
              "Numerics.hydro.hourglass.subzonal_pressure_model must be one of "
              "{\"linearized\", \"eos_lookup\"}");
        }
        if (hg.subzonal_pressure_model == "eos_lookup") {
          throw ConfigError(
              "Numerics.hydro.hourglass.subzonal_pressure_model=\"eos_lookup\" "
              "is reserved for a future implementation");
        }
      }
      if (has_key(hourglass, "max_force_per_node_fraction")) {
        hg.max_force_per_node_fraction = numeric_as_double(
            hourglass["max_force_per_node_fraction"],
            "Numerics.hydro.hourglass.max_force_per_node_fraction");
        ensure_positive(hg.max_force_per_node_fraction,
                        "Numerics.hydro.hourglass.max_force_per_node_fraction");
      }
    }
    if (has_key(hydro, "axis_projection")) {
      const py::handle axis_projection_obj = hydro["axis_projection"];
      if (!py::isinstance<py::dict>(axis_projection_obj)) {
        throw_value_type_error(
            "Numerics.hydro.axis_projection", "dict", axis_projection_obj);
      }
      const py::dict axis_projection =
          py::reinterpret_borrow<py::dict>(axis_projection_obj);
      enforce_known_keys(
          axis_projection,
          "Numerics.hydro.axis_projection",
          {"enabled", "shadow_only", "q_on", "q_floor", "patch_halfwidth",
           "log_every_n_steps"});
      auto& projection = numerics.hydro.axis_projection;
      if (has_key(axis_projection, "enabled")) {
        projection.enabled = strict_bool(
            axis_projection["enabled"],
            "Numerics.hydro.axis_projection.enabled");
      }
      if (has_key(axis_projection, "shadow_only")) {
        projection.shadow_only = strict_bool(
            axis_projection["shadow_only"],
            "Numerics.hydro.axis_projection.shadow_only");
      }
      if (has_key(axis_projection, "q_on")) {
        projection.q_on = numeric_as_double(
            axis_projection["q_on"],
            "Numerics.hydro.axis_projection.q_on");
      }
      if (!(std::isfinite(projection.q_on) &&
            projection.q_on > 0.0 && projection.q_on < 1.0)) {
        throw ValueError(
            "Numerics.hydro.axis_projection.q_on must be finite and in (0, 1)");
      }
      if (has_key(axis_projection, "q_floor")) {
        projection.q_floor = numeric_as_double(
            axis_projection["q_floor"],
            "Numerics.hydro.axis_projection.q_floor");
      }
      if (!(std::isfinite(projection.q_floor) &&
            projection.q_floor > 0.0 &&
            projection.q_floor < projection.q_on)) {
        throw ValueError(
            "Numerics.hydro.axis_projection.q_floor must be finite and in "
            "(0, q_on)");
      }
      if (has_key(axis_projection, "patch_halfwidth")) {
        projection.patch_halfwidth = strict_int32(
            axis_projection["patch_halfwidth"],
            "Numerics.hydro.axis_projection.patch_halfwidth");
      }
      if (projection.patch_halfwidth < 1 ||
          projection.patch_halfwidth > 8) {
        throw ValueError(
            "Numerics.hydro.axis_projection.patch_halfwidth must be in [1, 8]");
      }
      if (has_key(axis_projection, "log_every_n_steps")) {
        projection.log_every_n_steps = strict_int32(
            axis_projection["log_every_n_steps"],
            "Numerics.hydro.axis_projection.log_every_n_steps");
      }
      ensure_non_negative(
          projection.log_every_n_steps,
          "Numerics.hydro.axis_projection.log_every_n_steps");
    }
    if (has_key(hydro, "plasma_viscosity")) {
      const py::handle pv_obj = hydro["plasma_viscosity"];
      if (!py::isinstance<py::dict>(pv_obj)) {
        throw_value_type_error("Numerics.hydro.plasma_viscosity", "dict",
                               pv_obj);
      }
      const py::dict pv = py::reinterpret_borrow<py::dict>(pv_obj);
      enforce_known_keys(pv, "Numerics.hydro.plasma_viscosity",
                         {"enabled", "model", "species", "eta_const",
                          "eta0_scale", "mfp_cap_cells", "lnlambda_fixed",
                          "dt_safety"});
      auto& pvc = numerics.hydro.plasma_viscosity;
      if (has_key(pv, "enabled")) {
        pvc.enabled =
            strict_bool(pv["enabled"], "Numerics.hydro.plasma_viscosity.enabled");
      }
      if (has_key(pv, "model")) {
        pvc.model =
            strict_string(pv["model"], "Numerics.hydro.plasma_viscosity.model");
        if (pvc.model != "braginskii" && pvc.model != "constant") {
          throw ConfigError(
              "Numerics.hydro.plasma_viscosity.model must be"
              " \"braginskii\" or \"constant\"");
        }
      }
      if (has_key(pv, "species")) {
        pvc.species = strict_string(pv["species"],
                                    "Numerics.hydro.plasma_viscosity.species");
        if (pvc.species != "ion" && pvc.species != "electron" &&
            pvc.species != "both") {
          throw ConfigError(
              "Numerics.hydro.plasma_viscosity.species must be"
              " \"ion\", \"electron\", or \"both\"");
        }
      }
      if (has_key(pv, "eta_const")) {
        pvc.eta_const = numeric_as_double(
            pv["eta_const"], "Numerics.hydro.plasma_viscosity.eta_const");
        ensure_non_negative(pvc.eta_const,
                            "Numerics.hydro.plasma_viscosity.eta_const");
      }
      if (has_key(pv, "eta0_scale")) {
        pvc.eta0_scale = numeric_as_double(
            pv["eta0_scale"], "Numerics.hydro.plasma_viscosity.eta0_scale");
        ensure_positive(pvc.eta0_scale,
                        "Numerics.hydro.plasma_viscosity.eta0_scale");
      }
      if (has_key(pv, "mfp_cap_cells")) {
        pvc.mfp_cap_cells = numeric_as_double(
            pv["mfp_cap_cells"],
            "Numerics.hydro.plasma_viscosity.mfp_cap_cells");
        ensure_non_negative(pvc.mfp_cap_cells,
                            "Numerics.hydro.plasma_viscosity.mfp_cap_cells");
      }
      if (has_key(pv, "lnlambda_fixed")) {
        pvc.lnlambda_fixed = numeric_as_double(
            pv["lnlambda_fixed"],
            "Numerics.hydro.plasma_viscosity.lnlambda_fixed");
        ensure_non_negative(pvc.lnlambda_fixed,
                            "Numerics.hydro.plasma_viscosity.lnlambda_fixed");
      }
      if (has_key(pv, "dt_safety")) {
        pvc.dt_safety = numeric_as_double(
            pv["dt_safety"], "Numerics.hydro.plasma_viscosity.dt_safety");
        ensure_positive(pvc.dt_safety,
                        "Numerics.hydro.plasma_viscosity.dt_safety");
      }
    }
    if (has_key(hydro, "axis_motion_floor_fraction")) {
      numerics.hydro.axis_motion_floor_fraction = numeric_as_double(
          hydro["axis_motion_floor_fraction"],
          "Numerics.hydro.axis_motion_floor_fraction");
      if (!(numerics.hydro.axis_motion_floor_fraction >= 0.0 &&
            numerics.hydro.axis_motion_floor_fraction <= 1.0)) {
        throw ValueError("Numerics.hydro.axis_motion_floor_fraction must be in [0, 1]");
      }
    }
    if (has_key(hydro, "axis_margin_dt_floor_fraction")) {
      numerics.hydro.axis_margin_dt_floor_fraction = numeric_as_double(
          hydro["axis_margin_dt_floor_fraction"],
          "Numerics.hydro.axis_margin_dt_floor_fraction");
      if (!(numerics.hydro.axis_margin_dt_floor_fraction >= 0.0 &&
            numerics.hydro.axis_margin_dt_floor_fraction <= 1.0)) {
        throw ValueError("Numerics.hydro.axis_margin_dt_floor_fraction must be in [0, 1]");
      }
    }
    if (has_key(hydro, "volume_rate_cfl_enabled")) {
      numerics.hydro.volume_rate_cfl_enabled = strict_bool(
          hydro["volume_rate_cfl_enabled"],
          "Numerics.hydro.volume_rate_cfl_enabled");
    }
    if (has_key(hydro, "volume_rate_cfl_threshold")) {
      numerics.hydro.volume_rate_cfl_threshold = numeric_as_double(
          hydro["volume_rate_cfl_threshold"],
          "Numerics.hydro.volume_rate_cfl_threshold");
      if (!(numerics.hydro.volume_rate_cfl_threshold > 0.0)) {
        throw ValueError("Numerics.hydro.volume_rate_cfl_threshold must be > 0");
      }
    }
    if (has_key(hydro, "tri_fan_center_cfl_enabled")) {
      numerics.hydro.tri_fan_center_cfl_enabled = strict_bool(
          hydro["tri_fan_center_cfl_enabled"],
          "Numerics.hydro.tri_fan_center_cfl_enabled");
    }
    if (has_key(hydro, "tri_fan_center_cfl_safety")) {
      numerics.hydro.tri_fan_center_cfl_safety = numeric_as_double(
          hydro["tri_fan_center_cfl_safety"],
          "Numerics.hydro.tri_fan_center_cfl_safety");
      if (!(numerics.hydro.tri_fan_center_cfl_safety > 0.0)) {
        throw ValueError("Numerics.hydro.tri_fan_center_cfl_safety must be > 0");
      }
    }
    if (has_key(hydro, "tri_fan_center_cfl_band_radial_index")) {
      numerics.hydro.tri_fan_center_cfl_band_radial_index = strict_int32(
          hydro["tri_fan_center_cfl_band_radial_index"],
          "Numerics.hydro.tri_fan_center_cfl_band_radial_index");
      if (numerics.hydro.tri_fan_center_cfl_band_radial_index < 0) {
        throw ValueError(
            "Numerics.hydro.tri_fan_center_cfl_band_radial_index must be >= 0");
      }
    }
    if (has_key(hydro, "corner_j_predict_cfl_enabled")) {
      numerics.hydro.corner_j_predict_cfl_enabled = strict_bool(
          hydro["corner_j_predict_cfl_enabled"],
          "Numerics.hydro.corner_j_predict_cfl_enabled");
    }
    if (has_key(hydro, "corner_j_predict_cfl_safety")) {
      numerics.hydro.corner_j_predict_cfl_safety = numeric_as_double(
          hydro["corner_j_predict_cfl_safety"],
          "Numerics.hydro.corner_j_predict_cfl_safety");
      if (!(numerics.hydro.corner_j_predict_cfl_safety > 0.0 &&
            numerics.hydro.corner_j_predict_cfl_safety <= 1.0)) {
        throw ValueError(
            "Numerics.hydro.corner_j_predict_cfl_safety must be in (0, 1]");
      }
    }
    if (has_key(hydro, "corner_j_predict_floor_frac")) {
      numerics.hydro.corner_j_predict_floor_frac = numeric_as_double(
          hydro["corner_j_predict_floor_frac"],
          "Numerics.hydro.corner_j_predict_floor_frac");
      if (!(numerics.hydro.corner_j_predict_floor_frac > 0.0 &&
            numerics.hydro.corner_j_predict_floor_frac <= 1.0)) {
        throw ValueError(
            "Numerics.hydro.corner_j_predict_floor_frac must be in (0, 1]");
      }
    }
    if (has_key(hydro, "corner_j_predict_max_shrink")) {
      numerics.hydro.corner_j_predict_max_shrink = numeric_as_double(
          hydro["corner_j_predict_max_shrink"],
          "Numerics.hydro.corner_j_predict_max_shrink");
      if (!(numerics.hydro.corner_j_predict_max_shrink > 0.0 &&
            numerics.hydro.corner_j_predict_max_shrink < 1.0)) {
        throw ValueError(
            "Numerics.hydro.corner_j_predict_max_shrink must be in (0, 1)");
      }
    }
    if (has_key(hydro, "corner_j_predict_shell_rings")) {
      numerics.hydro.corner_j_predict_shell_rings = strict_int32(
          hydro["corner_j_predict_shell_rings"],
          "Numerics.hydro.corner_j_predict_shell_rings");
      if (numerics.hydro.corner_j_predict_shell_rings < 0) {
        throw ValueError(
            "Numerics.hydro.corner_j_predict_shell_rings must be >= 0");
      }
    }
    if (has_key(hydro, "tri_fan_center_perturbation_diag_enabled")) {
      numerics.hydro.tri_fan_center_perturbation_diag_enabled = strict_bool(
          hydro["tri_fan_center_perturbation_diag_enabled"],
          "Numerics.hydro.tri_fan_center_perturbation_diag_enabled");
    }
    const bool center_cfl_scope_explicit =
        has_key(hydro, "center_cfl_scope");
    if (center_cfl_scope_explicit) {
      numerics.hydro.center_cfl_scope = parse_center_cfl_scope(
          strict_string(hydro["center_cfl_scope"],
                        "Numerics.hydro.center_cfl_scope"),
          "Numerics.hydro.center_cfl_scope");
    } else if (numerics.hydro.tri_fan_center_cfl_enabled) {
      numerics.hydro.center_cfl_scope =
          CenterCflScope::TRI_FAN_RADIAL_INDEX;
    }
    if (center_cfl_scope_explicit &&
        numerics.hydro.tri_fan_center_cfl_enabled &&
        numerics.hydro.center_cfl_scope !=
            CenterCflScope::TRI_FAN_RADIAL_INDEX) {
      throw ConfigError(
          "conflicting center-CFL scope and tri_fan_center_cfl_enabled");
    }
    const bool center_perturbation_diag_scope_explicit =
        has_key(hydro, "center_perturbation_diag_scope");
    if (center_perturbation_diag_scope_explicit) {
      numerics.hydro.center_perturbation_diag_scope =
          parse_center_perturbation_diag_scope(
              strict_string(hydro["center_perturbation_diag_scope"],
                            "Numerics.hydro.center_perturbation_diag_scope"),
              "Numerics.hydro.center_perturbation_diag_scope");
    } else if (numerics.hydro.tri_fan_center_perturbation_diag_enabled) {
      numerics.hydro.center_perturbation_diag_scope =
          CenterPerturbationDiagScope::TRI_FAN_FIRST_RING;
    }
    if (center_perturbation_diag_scope_explicit &&
        numerics.hydro.tri_fan_center_perturbation_diag_enabled &&
        numerics.hydro.center_perturbation_diag_scope !=
            CenterPerturbationDiagScope::TRI_FAN_FIRST_RING) {
      throw ConfigError(
          "conflicting center perturbation diagnostic scope and "
          "tri_fan_center_perturbation_diag_enabled");
    }
    if (has_key(hydro, "center_perturbation_diag_radial_bins")) {
      numerics.hydro.center_perturbation_diag_radial_bins = strict_int32(
          hydro["center_perturbation_diag_radial_bins"],
          "Numerics.hydro.center_perturbation_diag_radial_bins");
    }
    if (has_key(hydro, "rz_geometric_cfl_enabled")) {
      numerics.hydro.rz_geometric_cfl_enabled = strict_bool(
          hydro["rz_geometric_cfl_enabled"],
          "Numerics.hydro.rz_geometric_cfl_enabled");
    }
    if (has_key(hydro, "rz_geometric_cfl_etaV")) {
      numerics.hydro.rz_geometric_cfl_etaV = numeric_as_double(
          hydro["rz_geometric_cfl_etaV"],
          "Numerics.hydro.rz_geometric_cfl_etaV");
      if (!(numerics.hydro.rz_geometric_cfl_etaV > 0.0 &&
            numerics.hydro.rz_geometric_cfl_etaV <= 1.0)) {
        throw ValueError("Numerics.hydro.rz_geometric_cfl_etaV must be in (0, 1]");
      }
    }
    if (has_key(hydro, "rz_geometric_cfl_r_floor")) {
      numerics.hydro.rz_geometric_cfl_r_floor = numeric_as_double(
          hydro["rz_geometric_cfl_r_floor"],
          "Numerics.hydro.rz_geometric_cfl_r_floor");
      if (!(numerics.hydro.rz_geometric_cfl_r_floor >= 0.0)) {
        throw ValueError("Numerics.hydro.rz_geometric_cfl_r_floor must be >= 0");
      }
    }
    if (has_key(hydro, "rz_geometric_cfl_cumulative_protection_enabled")) {
      numerics.hydro.rz_geometric_cfl_cumulative_protection_enabled = strict_bool(
          hydro["rz_geometric_cfl_cumulative_protection_enabled"],
          "Numerics.hydro.rz_geometric_cfl_cumulative_protection_enabled");
    }
    if (has_key(hydro, "rz_geometric_cfl_v_initial_floor")) {
      numerics.hydro.rz_geometric_cfl_v_initial_floor = numeric_as_double(
          hydro["rz_geometric_cfl_v_initial_floor"],
          "Numerics.hydro.rz_geometric_cfl_v_initial_floor");
      if (!(numerics.hydro.rz_geometric_cfl_v_initial_floor >= 0.0 &&
            numerics.hydro.rz_geometric_cfl_v_initial_floor <= 1.0)) {
        throw ValueError(
            "Numerics.hydro.rz_geometric_cfl_v_initial_floor must be in [0, 1]");
      }
    }
    if (has_key(hydro, "rz_geometric_cfl_precise_u_half_enabled")) {
      numerics.hydro.rz_geometric_cfl_precise_u_half_enabled = strict_bool(
          hydro["rz_geometric_cfl_precise_u_half_enabled"],
          "Numerics.hydro.rz_geometric_cfl_precise_u_half_enabled");
    }
    if (has_key(hydro, "trial_volume_cfl_enabled")) {
      numerics.hydro.trial_volume_cfl_enabled = strict_bool(
          hydro["trial_volume_cfl_enabled"],
          "Numerics.hydro.trial_volume_cfl_enabled");
    }
    if (has_key(hydro, "trial_volume_cfl_floor_fraction")) {
      numerics.hydro.trial_volume_cfl_floor_fraction = numeric_as_double(
          hydro["trial_volume_cfl_floor_fraction"],
          "Numerics.hydro.trial_volume_cfl_floor_fraction");
      if (!(numerics.hydro.trial_volume_cfl_floor_fraction > 0.0 &&
            numerics.hydro.trial_volume_cfl_floor_fraction <= 1.0)) {
        throw ValueError(
            "Numerics.hydro.trial_volume_cfl_floor_fraction must be in (0, 1]");
      }
    }
    if (has_key(hydro, "trial_volume_cfl_shrink_fraction")) {
      numerics.hydro.trial_volume_cfl_shrink_fraction = numeric_as_double(
          hydro["trial_volume_cfl_shrink_fraction"],
          "Numerics.hydro.trial_volume_cfl_shrink_fraction");
      if (!(numerics.hydro.trial_volume_cfl_shrink_fraction > 0.0 &&
            numerics.hydro.trial_volume_cfl_shrink_fraction < 1.0)) {
        throw ValueError(
            "Numerics.hydro.trial_volume_cfl_shrink_fraction must be in (0, 1)");
      }
    }
    if (has_key(hydro, "corner_jacobian_ale_trigger_enabled")) {
      numerics.hydro.corner_jacobian_ale_trigger_enabled = strict_bool(
          hydro["corner_jacobian_ale_trigger_enabled"],
          "Numerics.hydro.corner_jacobian_ale_trigger_enabled");
    }
    if (has_key(hydro, "corner_jacobian_floor_eps")) {
      numerics.hydro.corner_jacobian_floor_eps = numeric_as_double(
          hydro["corner_jacobian_floor_eps"],
          "Numerics.hydro.corner_jacobian_floor_eps");
      if (!(numerics.hydro.corner_jacobian_floor_eps >= 0.0 &&
            numerics.hydro.corner_jacobian_floor_eps < 1.0)) {
        throw ValueError(
            "Numerics.hydro.corner_jacobian_floor_eps must be in [0, 1)");
      }
    }
    if (has_key(hydro, "corner_jacobian_ale_trigger_scale")) {
      numerics.hydro.corner_jacobian_ale_trigger_scale = numeric_as_double(
          hydro["corner_jacobian_ale_trigger_scale"],
          "Numerics.hydro.corner_jacobian_ale_trigger_scale");
      if (!(numerics.hydro.corner_jacobian_ale_trigger_scale > 0.0 &&
            numerics.hydro.corner_jacobian_ale_trigger_scale <= 1.0)) {
        throw ValueError(
            "Numerics.hydro.corner_jacobian_ale_trigger_scale must be in (0, 1]");
      }
    }
    if (has_key(hydro, "in_hydro_corner_j_guard_enabled")) {
      numerics.hydro.in_hydro_corner_j_guard_enabled = strict_bool(
          hydro["in_hydro_corner_j_guard_enabled"],
          "Numerics.hydro.in_hydro_corner_j_guard_enabled");
    }
    if (has_key(hydro, "in_hydro_gauss_j_guard_enabled")) {
      numerics.hydro.in_hydro_gauss_j_guard_enabled = strict_bool(
          hydro["in_hydro_gauss_j_guard_enabled"],
          "Numerics.hydro.in_hydro_gauss_j_guard_enabled");
    }
    if (has_key(hydro, "in_hydro_rz_volume_guard_enabled")) {
      numerics.hydro.in_hydro_rz_volume_guard_enabled = strict_bool(
          hydro["in_hydro_rz_volume_guard_enabled"],
          "Numerics.hydro.in_hydro_rz_volume_guard_enabled");
    }
    if (has_key(hydro, "in_hydro_gauss_j_floor_rel")) {
      numerics.hydro.in_hydro_gauss_j_floor_rel = numeric_as_double(
          hydro["in_hydro_gauss_j_floor_rel"],
          "Numerics.hydro.in_hydro_gauss_j_floor_rel");
      if (!(numerics.hydro.in_hydro_gauss_j_floor_rel > 0.0)) {
        throw ValueError("Numerics.hydro.in_hydro_gauss_j_floor_rel must be > 0");
      }
    }
    if (has_key(hydro, "in_hydro_rz_volume_floor_rel")) {
      numerics.hydro.in_hydro_rz_volume_floor_rel = numeric_as_double(
          hydro["in_hydro_rz_volume_floor_rel"],
          "Numerics.hydro.in_hydro_rz_volume_floor_rel");
      if (!(numerics.hydro.in_hydro_rz_volume_floor_rel > 0.0)) {
        throw ValueError(
            "Numerics.hydro.in_hydro_rz_volume_floor_rel must be > 0");
      }
    }
    if (has_key(hydro, "mesh_quality_dt_cfl_enabled")) {
      numerics.hydro.mesh_quality_dt_cfl_enabled = strict_bool(
          hydro["mesh_quality_dt_cfl_enabled"],
          "Numerics.hydro.mesh_quality_dt_cfl_enabled");
    }
    if (has_key(hydro, "mesh_quality_dt_safety_alpha")) {
      numerics.hydro.mesh_quality_dt_safety_alpha = numeric_as_double(
          hydro["mesh_quality_dt_safety_alpha"],
          "Numerics.hydro.mesh_quality_dt_safety_alpha");
      if (!(numerics.hydro.mesh_quality_dt_safety_alpha > 0.0 &&
            numerics.hydro.mesh_quality_dt_safety_alpha <= 1.0)) {
        throw ValueError(
            "Numerics.hydro.mesh_quality_dt_safety_alpha must be in (0, 1]");
      }
    }
    if (has_key(hydro, "mesh_quality_dt_corner_j_enabled")) {
      numerics.hydro.mesh_quality_dt_corner_j_enabled = strict_bool(
          hydro["mesh_quality_dt_corner_j_enabled"],
          "Numerics.hydro.mesh_quality_dt_corner_j_enabled");
    }
    if (has_key(hydro, "mesh_quality_dt_gauss_j_enabled")) {
      numerics.hydro.mesh_quality_dt_gauss_j_enabled = strict_bool(
          hydro["mesh_quality_dt_gauss_j_enabled"],
          "Numerics.hydro.mesh_quality_dt_gauss_j_enabled");
    }
    if (has_key(hydro, "mesh_quality_dt_rz_volume_enabled")) {
      numerics.hydro.mesh_quality_dt_rz_volume_enabled = strict_bool(
          hydro["mesh_quality_dt_rz_volume_enabled"],
          "Numerics.hydro.mesh_quality_dt_rz_volume_enabled");
    }
    if (has_key(hydro, "mesh_quality_dt_axis_margin_additive")) {
      numerics.hydro.mesh_quality_dt_axis_margin_additive = strict_bool(
          hydro["mesh_quality_dt_axis_margin_additive"],
          "Numerics.hydro.mesh_quality_dt_axis_margin_additive");
    }
    if (has_key(hydro, "mesh_quality_dt_corner_j_floor_rel")) {
      numerics.hydro.mesh_quality_dt_corner_j_floor_rel = numeric_as_double(
          hydro["mesh_quality_dt_corner_j_floor_rel"],
          "Numerics.hydro.mesh_quality_dt_corner_j_floor_rel");
      if (!(numerics.hydro.mesh_quality_dt_corner_j_floor_rel > 0.0)) {
        throw ValueError(
            "Numerics.hydro.mesh_quality_dt_corner_j_floor_rel must be > 0");
      }
    }
    if (has_key(hydro, "mesh_quality_dt_gauss_j_floor_rel")) {
      numerics.hydro.mesh_quality_dt_gauss_j_floor_rel = numeric_as_double(
          hydro["mesh_quality_dt_gauss_j_floor_rel"],
          "Numerics.hydro.mesh_quality_dt_gauss_j_floor_rel");
      if (!(numerics.hydro.mesh_quality_dt_gauss_j_floor_rel > 0.0)) {
        throw ValueError(
            "Numerics.hydro.mesh_quality_dt_gauss_j_floor_rel must be > 0");
      }
    }
    if (has_key(hydro, "mesh_quality_dt_rz_volume_floor_rel")) {
      numerics.hydro.mesh_quality_dt_rz_volume_floor_rel = numeric_as_double(
          hydro["mesh_quality_dt_rz_volume_floor_rel"],
          "Numerics.hydro.mesh_quality_dt_rz_volume_floor_rel");
      if (!(numerics.hydro.mesh_quality_dt_rz_volume_floor_rel > 0.0)) {
        throw ValueError(
            "Numerics.hydro.mesh_quality_dt_rz_volume_floor_rel must be > 0");
      }
    }
    if (has_key(hydro, "ring7_quotient_enabled")) {
      numerics.hydro.ring7_quotient_enabled = strict_bool(
          hydro["ring7_quotient_enabled"],
          "Numerics.hydro.ring7_quotient_enabled");
    }
    if (has_key(hydro, "regime_aware_corner_j_guard_enabled")) {
      numerics.hydro.regime_aware_corner_j_guard_enabled = strict_bool(
          hydro["regime_aware_corner_j_guard_enabled"],
          "Numerics.hydro.regime_aware_corner_j_guard_enabled");
    }
    if (has_key(hydro, "axis_margin_guard_enabled")) {
      numerics.hydro.axis_margin_guard_enabled = strict_bool(
          hydro["axis_margin_guard_enabled"],
          "Numerics.hydro.axis_margin_guard_enabled");
    }
    if (has_key(hydro, "axis_margin_additive_in_action8_enabled")) {
      numerics.hydro.axis_margin_additive_in_action8_enabled = strict_bool(
          hydro["axis_margin_additive_in_action8_enabled"],
          "Numerics.hydro.axis_margin_additive_in_action8_enabled");
    }
    if (has_key(hydro, "axis_guard_band_cells")) {
      numerics.hydro.axis_guard_band_cells = strict_int32(
          hydro["axis_guard_band_cells"],
          "Numerics.hydro.axis_guard_band_cells");
      if (numerics.hydro.axis_guard_band_cells < 0) {
        throw ValueError("Numerics.hydro.axis_guard_band_cells must be >= 0");
      }
    }
    if (has_key(hydro, "driver_full_step_retry_enabled")) {
      numerics.hydro.driver_full_step_retry_enabled = strict_bool(
          hydro["driver_full_step_retry_enabled"],
          "Numerics.hydro.driver_full_step_retry_enabled");
    }
    if (has_key(hydro, "driver_full_step_retry_max_attempts")) {
      numerics.hydro.driver_full_step_retry_max_attempts = strict_int32(
          hydro["driver_full_step_retry_max_attempts"],
          "Numerics.hydro.driver_full_step_retry_max_attempts");
      if (numerics.hydro.driver_full_step_retry_max_attempts < 0) {
        throw ValueError(
            "Numerics.hydro.driver_full_step_retry_max_attempts must be >= 0");
      }
    }
    if (has_key(hydro, "dispatcher_state_sensitive_bypass_enabled")) {
      numerics.hydro.dispatcher_state_sensitive_bypass_enabled = strict_bool(
          hydro["dispatcher_state_sensitive_bypass_enabled"],
          "Numerics.hydro.dispatcher_state_sensitive_bypass_enabled");
    }
    if (has_key(hydro, "dispatcher_state_sensitive_repair_cap_per_step")) {
      numerics.hydro.dispatcher_state_sensitive_repair_cap_per_step =
          strict_int32(
              hydro["dispatcher_state_sensitive_repair_cap_per_step"],
              "Numerics.hydro.dispatcher_state_sensitive_repair_cap_per_step");
      if (numerics.hydro.dispatcher_state_sensitive_repair_cap_per_step < 1) {
        throw ValueError(
            "Numerics.hydro.dispatcher_state_sensitive_repair_cap_per_step "
            "must be >= 1");
      }
    }
    if (has_key(hydro, "strategy_first_retry_enabled")) {
      numerics.hydro.strategy_first_retry_enabled = strict_bool(
          hydro["strategy_first_retry_enabled"],
          "Numerics.hydro.strategy_first_retry_enabled");
    }
    if (has_key(hydro, "strategy_first_max_same_dt_attempts")) {
      numerics.hydro.strategy_first_max_same_dt_attempts = strict_int32(
          hydro["strategy_first_max_same_dt_attempts"],
          "Numerics.hydro.strategy_first_max_same_dt_attempts");
      if (numerics.hydro.strategy_first_max_same_dt_attempts < 0) {
        throw ValueError(
            "Numerics.hydro.strategy_first_max_same_dt_attempts must be >= 0");
      }
    }
    if (has_key(hydro, "driver_retry_active_mesh_repair_enabled")) {
      numerics.hydro.driver_retry_active_mesh_repair_enabled = strict_bool(
          hydro["driver_retry_active_mesh_repair_enabled"],
          "Numerics.hydro.driver_retry_active_mesh_repair_enabled");
    }
    if (has_key(hydro, "driver_retry_corner_balance_threshold")) {
      numerics.hydro.driver_retry_corner_balance_threshold = numeric_as_double(
          hydro["driver_retry_corner_balance_threshold"],
          "Numerics.hydro.driver_retry_corner_balance_threshold");
      if (!(numerics.hydro.driver_retry_corner_balance_threshold > 0.0 &&
            numerics.hydro.driver_retry_corner_balance_threshold < 1.0)) {
        throw ValueError(
            "Numerics.hydro.driver_retry_corner_balance_threshold must be in (0, 1)");
      }
    }
    if (has_key(hydro, "cascade_on_hydro_retry_enabled")) {
      numerics.hydro.cascade_on_hydro_retry_enabled = strict_bool(
          hydro["cascade_on_hydro_retry_enabled"],
          "Numerics.hydro.cascade_on_hydro_retry_enabled");
    }
    if (has_key(hydro, "driver_retry_use_suggested_dt_enabled")) {
      numerics.hydro.driver_retry_use_suggested_dt_enabled = strict_bool(
          hydro["driver_retry_use_suggested_dt_enabled"],
          "Numerics.hydro.driver_retry_use_suggested_dt_enabled");
    }
    if (has_key(hydro, "geometric_retry_stagnation")) {
      const py::handle stagnation_obj = hydro["geometric_retry_stagnation"];
      if (!py::isinstance<py::dict>(stagnation_obj)) {
        throw_value_type_error("Numerics.hydro.geometric_retry_stagnation",
                               "dict",
                               stagnation_obj);
      }
      const py::dict stagnation = py::reinterpret_borrow<py::dict>(stagnation_obj);
      enforce_known_keys(stagnation,
                         "Numerics.hydro.geometric_retry_stagnation",
                         {"enabled",
                          "same_cell_count_threshold",
                          "sigma_rel_tol",
                          "dt_drop_factor",
                          "force_diagnostic_dump"});
      auto& gs = numerics.hydro.geometric_retry_stagnation;
      if (has_key(stagnation, "enabled")) {
        gs.enabled = strict_bool(
            stagnation["enabled"],
            "Numerics.hydro.geometric_retry_stagnation.enabled");
      }
      if (has_key(stagnation, "same_cell_count_threshold")) {
        gs.same_cell_count_threshold = strict_int32(
            stagnation["same_cell_count_threshold"],
            "Numerics.hydro.geometric_retry_stagnation."
            "same_cell_count_threshold");
        if (gs.same_cell_count_threshold < 1) {
          throw ValueError(
              "Numerics.hydro.geometric_retry_stagnation."
              "same_cell_count_threshold must be >= 1");
        }
      }
      if (has_key(stagnation, "sigma_rel_tol")) {
        gs.sigma_rel_tol = numeric_as_double(
            stagnation["sigma_rel_tol"],
            "Numerics.hydro.geometric_retry_stagnation.sigma_rel_tol");
        if (!(gs.sigma_rel_tol > 0.0 && gs.sigma_rel_tol <= 1.0)) {
          throw ValueError(
              "Numerics.hydro.geometric_retry_stagnation.sigma_rel_tol "
              "must be in (0, 1]");
        }
      }
      if (has_key(stagnation, "dt_drop_factor")) {
        gs.dt_drop_factor = numeric_as_double(
            stagnation["dt_drop_factor"],
            "Numerics.hydro.geometric_retry_stagnation.dt_drop_factor");
        if (!(gs.dt_drop_factor > 0.0 && gs.dt_drop_factor < 1.0)) {
          throw ValueError(
              "Numerics.hydro.geometric_retry_stagnation.dt_drop_factor "
              "must be in (0, 1)");
        }
      }
      if (has_key(stagnation, "force_diagnostic_dump")) {
        gs.force_diagnostic_dump = strict_bool(
            stagnation["force_diagnostic_dump"],
            "Numerics.hydro.geometric_retry_stagnation.force_diagnostic_dump");
      }
    }
    if (has_key(hydro, "mesh_geometry_soft_fail_enabled")) {
      numerics.hydro.mesh_geometry_soft_fail_enabled = strict_bool(
          hydro["mesh_geometry_soft_fail_enabled"],
          "Numerics.hydro.mesh_geometry_soft_fail_enabled");
    }
    if (has_key(hydro, "T_start_eV")) {
      hydro_t_start_eV = numeric_as_double(hydro["T_start_eV"], "Numerics.hydro.T_start_eV");
      ensure_non_negative(hydro_t_start_eV, "Numerics.hydro.T_start_eV");
      numerics.T_start_eV = hydro_t_start_eV;
      if (has_key(kwargs, "T_start_eV")) {
        tenryu::core::log_warning(
            "Numerics.hydro.T_start_eV overrides deprecated Numerics.T_start_eV");
      }
    }
    if (has_key(hydro, "boundary")) {
      const py::handle boundary_obj = hydro["boundary"];
      if (py::isinstance<py::str>(boundary_obj)) {
        numerics.hydro.boundary_1d =
            normalize_boundary_mode(
                strict_string(boundary_obj, "Numerics.hydro.boundary"),
                "Numerics.hydro.boundary");
        if (!is_hydro_boundary(numerics.hydro.boundary_1d)) {
          throw ValueError(
              "Numerics.hydro.boundary must be one of {\"free\", \"fixed\", \"reflect\", \"pressure\", \"axis\"}, got " +
              numerics.hydro.boundary_1d);
        }
      } else if (py::isinstance<py::dict>(boundary_obj)) {
        const py::dict boundary = py::reinterpret_borrow<py::dict>(boundary_obj);
        enforce_known_keys(boundary, "Numerics.hydro.boundary",
                           {"r_inner", "r_outer", "z_bottom", "z_top",
                            "mesh_tangential_target", "state_supply_donor_mode"});
        if (has_key(boundary, "r_inner")) {
          numerics.hydro.boundary_2d.r_inner =
              normalize_boundary_mode(
                  strict_string(boundary["r_inner"], "Numerics.hydro.boundary.r_inner"),
                  "Numerics.hydro.boundary.r_inner");
          if (!is_hydro_boundary(numerics.hydro.boundary_2d.r_inner) &&
              numerics.hydro.boundary_2d.r_inner != "pinned") {
            throw ValueError(
                "Numerics.hydro.boundary.r_inner must be one of {\"free\", \"fixed\", \"reflect\", \"pressure\", \"axis\", \"pinned\"}, got " +
                numerics.hydro.boundary_2d.r_inner);
          }
        }
        if (has_key(boundary, "r_outer")) {
          numerics.hydro.boundary_2d.r_outer =
              normalize_boundary_mode(
                  strict_string(boundary["r_outer"], "Numerics.hydro.boundary.r_outer"),
                  "Numerics.hydro.boundary.r_outer");
          if (!is_hydro_boundary(numerics.hydro.boundary_2d.r_outer)) {
            throw ValueError(
                "Numerics.hydro.boundary.r_outer must be one of {\"free\", \"fixed\", \"reflect\", \"pressure\", \"axis\"}, got " +
                numerics.hydro.boundary_2d.r_outer);
          }
        }
        if (has_key(boundary, "z_bottom")) {
          parse_hydro_z_boundary(numerics.hydro.boundary_2d.z_bottom_cfg,
                                 numerics.hydro.boundary_2d.z_bottom,
                                 boundary["z_bottom"],
                                 "Numerics.hydro.boundary.z_bottom");
        }
        if (has_key(boundary, "z_top")) {
          parse_hydro_z_boundary(numerics.hydro.boundary_2d.z_top_cfg,
                                 numerics.hydro.boundary_2d.z_top,
                                 boundary["z_top"],
                                 "Numerics.hydro.boundary.z_top");
        }
        if (has_key(boundary, "mesh_tangential_target")) {
          numerics.hydro.boundary_2d.mesh_tangential_target = strict_string(
              boundary["mesh_tangential_target"],
              "Numerics.hydro.boundary.mesh_tangential_target");
          if (!is_mesh_tangential_target(
                  numerics.hydro.boundary_2d.mesh_tangential_target)) {
            throw ValueError(
                "Numerics.hydro.boundary.mesh_tangential_target must be one of "
                "{\"lagrangian\", \"reference\"}, got " +
                numerics.hydro.boundary_2d.mesh_tangential_target);
          }
        }
        if (has_key(boundary, "state_supply_donor_mode")) {
          numerics.hydro.boundary_2d.state_supply_donor_mode = strict_string(
              boundary["state_supply_donor_mode"],
              "Numerics.hydro.boundary.state_supply_donor_mode");
          if (!is_state_supply_donor_mode(
                  numerics.hydro.boundary_2d.state_supply_donor_mode)) {
            throw ValueError(
                "Numerics.hydro.boundary.state_supply_donor_mode must be one of "
                "{\"interior_per_i\", \"interior_radial_average\"}, got " +
                numerics.hydro.boundary_2d.state_supply_donor_mode);
          }
        }
        numerics.hydro.boundary_2d.sync_legacy_strings();
      } else {
        throw_value_type_error("Numerics.hydro.boundary", "str|dict", boundary_obj);
      }
    }
    if (has_key(hydro, "boundary_1d")) {
      numerics.hydro.boundary_1d =
          normalize_boundary_mode(
              strict_string(hydro["boundary_1d"], "Numerics.hydro.boundary_1d"),
              "Numerics.hydro.boundary_1d");
      if (!is_hydro_boundary(numerics.hydro.boundary_1d)) {
        throw ValueError(
            "Numerics.hydro.boundary_1d must be one of {\"free\", \"fixed\", \"reflect\", \"pressure\", \"axis\"}, got " +
            numerics.hydro.boundary_1d);
      }
    }
    if (has_key(hydro, "boundary_2d")) {
      const py::handle boundary2d_obj = hydro["boundary_2d"];
      if (!py::isinstance<py::dict>(boundary2d_obj)) {
        throw_value_type_error("Numerics.hydro.boundary_2d", "dict", boundary2d_obj);
      }
      const py::dict boundary2d = py::reinterpret_borrow<py::dict>(boundary2d_obj);
      enforce_known_keys(boundary2d, "Numerics.hydro.boundary_2d",
                         {"r_inner", "r_outer", "z_bottom", "z_top",
                          "mesh_tangential_target", "state_supply_donor_mode"});
      if (has_key(boundary2d, "r_inner")) {
        numerics.hydro.boundary_2d.r_inner =
            normalize_boundary_mode(
                strict_string(boundary2d["r_inner"], "Numerics.hydro.boundary_2d.r_inner"),
                "Numerics.hydro.boundary_2d.r_inner");
        if (!is_hydro_boundary(numerics.hydro.boundary_2d.r_inner) &&
            numerics.hydro.boundary_2d.r_inner != "pinned") {
          throw ValueError(
              "Numerics.hydro.boundary_2d.r_inner must be one of {\"free\", \"fixed\", \"reflect\", \"pressure\", \"axis\", \"pinned\"}, got " +
              numerics.hydro.boundary_2d.r_inner);
        }
      }
      if (has_key(boundary2d, "r_outer")) {
        numerics.hydro.boundary_2d.r_outer =
            normalize_boundary_mode(
                strict_string(boundary2d["r_outer"], "Numerics.hydro.boundary_2d.r_outer"),
                "Numerics.hydro.boundary_2d.r_outer");
        if (!is_hydro_boundary(numerics.hydro.boundary_2d.r_outer)) {
          throw ValueError(
              "Numerics.hydro.boundary_2d.r_outer must be one of {\"free\", \"fixed\", \"reflect\", \"pressure\", \"axis\"}, got " +
              numerics.hydro.boundary_2d.r_outer);
        }
      }
      if (has_key(boundary2d, "z_bottom")) {
        parse_hydro_z_boundary(numerics.hydro.boundary_2d.z_bottom_cfg,
                               numerics.hydro.boundary_2d.z_bottom,
                               boundary2d["z_bottom"],
                               "Numerics.hydro.boundary_2d.z_bottom");
      }
      if (has_key(boundary2d, "z_top")) {
        parse_hydro_z_boundary(numerics.hydro.boundary_2d.z_top_cfg,
                               numerics.hydro.boundary_2d.z_top,
                               boundary2d["z_top"],
                               "Numerics.hydro.boundary_2d.z_top");
      }
      if (has_key(boundary2d, "mesh_tangential_target")) {
        numerics.hydro.boundary_2d.mesh_tangential_target = strict_string(
            boundary2d["mesh_tangential_target"],
            "Numerics.hydro.boundary_2d.mesh_tangential_target");
        if (!is_mesh_tangential_target(
                numerics.hydro.boundary_2d.mesh_tangential_target)) {
          throw ValueError(
              "Numerics.hydro.boundary_2d.mesh_tangential_target must be one of "
              "{\"lagrangian\", \"reference\"}, got " +
              numerics.hydro.boundary_2d.mesh_tangential_target);
        }
      }
      if (has_key(boundary2d, "state_supply_donor_mode")) {
        numerics.hydro.boundary_2d.state_supply_donor_mode = strict_string(
            boundary2d["state_supply_donor_mode"],
            "Numerics.hydro.boundary_2d.state_supply_donor_mode");
        if (!is_state_supply_donor_mode(
                numerics.hydro.boundary_2d.state_supply_donor_mode)) {
          throw ValueError(
              "Numerics.hydro.boundary_2d.state_supply_donor_mode must be one of "
              "{\"interior_per_i\", \"interior_radial_average\"}, got " +
              numerics.hydro.boundary_2d.state_supply_donor_mode);
        }
      }
      numerics.hydro.boundary_2d.sync_legacy_strings();
    }
    const bool av_c1_explicit =
        has_key(hydro, "av_C1") || has_key(hydro, "av_linear");
    const bool av_c2_explicit =
        has_key(hydro, "av_C2") || has_key(hydro, "av_quadratic");
    if (has_key(hydro, "av_C1")) {
      numerics.hydro.av_linear =
          numeric_as_double(hydro["av_C1"], "Numerics.hydro.av_C1");
    }
    if (has_key(hydro, "av_C2")) {
      numerics.hydro.av_quadratic =
          numeric_as_double(hydro["av_C2"], "Numerics.hydro.av_C2");
    }
    if (has_key(hydro, "av_linear")) {
      numerics.hydro.av_linear =
          numeric_as_double(hydro["av_linear"], "Numerics.hydro.av_linear");
    }
    if (has_key(hydro, "av_quadratic")) {
      numerics.hydro.av_quadratic =
          numeric_as_double(hydro["av_quadratic"], "Numerics.hydro.av_quadratic");
    }
    if (has_key(hydro, "csw98_degenerate_side_floor_rel")) {
      numerics.hydro.csw98_degenerate_side_floor_rel = numeric_as_double(
          hydro["csw98_degenerate_side_floor_rel"],
          "Numerics.hydro.csw98_degenerate_side_floor_rel");
    }
    if (has_key(hydro, "csw98_damper_impulse_beta")) {
      numerics.hydro.csw98_damper_impulse_beta = numeric_as_double(
          hydro["csw98_damper_impulse_beta"],
          "Numerics.hydro.csw98_damper_impulse_beta");
    }
    if (has_key(hydro, "csw98_axisline_av_mode")) {
      numerics.hydro.csw98_axisline_av_mode = strict_string(
          hydro["csw98_axisline_av_mode"],
          "Numerics.hydro.csw98_axisline_av_mode");
      if (numerics.hydro.csw98_axisline_av_mode != "off" &&
          numerics.hydro.csw98_axisline_av_mode != "d1prime") {
        throw ValueError(
            "Numerics.hydro.csw98_axisline_av_mode must be one of {\"off\", \"d1prime\"}");
      }
    }
    if (has_key(hydro, "csw98_axisline_d1prime_cfl_enabled")) {
      numerics.hydro.csw98_axisline_d1prime_cfl_enabled = strict_bool(
          hydro["csw98_axisline_d1prime_cfl_enabled"],
          "Numerics.hydro.csw98_axisline_d1prime_cfl_enabled");
    }
    if (has_key(hydro, "csw98_limiter_shock_floor_enabled")) {
      numerics.hydro.csw98_limiter_shock_floor_enabled = strict_bool(
          hydro["csw98_limiter_shock_floor_enabled"],
          "Numerics.hydro.csw98_limiter_shock_floor_enabled");
    }
    if (has_key(hydro, "csw98_axisline_work_planar_enabled")) {
      numerics.hydro.csw98_axisline_work_planar_enabled = strict_bool(
          hydro["csw98_axisline_work_planar_enabled"],
          "Numerics.hydro.csw98_axisline_work_planar_enabled");
    }
    if (has_key(hydro, "tensor_av_C1")) {
      numerics.hydro.tensor_av_C1 = numeric_as_double(
          hydro["tensor_av_C1"], "Numerics.hydro.tensor_av_C1");
    }
    if (has_key(hydro, "tensor_av_C2")) {
      numerics.hydro.tensor_av_C2 = numeric_as_double(
          hydro["tensor_av_C2"], "Numerics.hydro.tensor_av_C2");
    }
    if (numerics.hydro.av_model == AvModel::CswEdge ||
        numerics.hydro.av_model == AvModel::CswEdgeCsw98) {
      if (!av_c1_explicit) {
        numerics.hydro.av_linear = 1.0;
      }
      if (!av_c2_explicit) {
        numerics.hydro.av_quadratic = 1.0;
      }
    }
    if (has_key(hydro, "av_qcap_over_p")) {
      numerics.hydro.av_qcap_over_p =
          numeric_as_double(hydro["av_qcap_over_p"],
                            "Numerics.hydro.av_qcap_over_p");
      ensure_non_negative(numerics.hydro.av_qcap_over_p,
                          "Numerics.hydro.av_qcap_over_p");
    }
    if (has_key(hydro, "av_qcap_center_band_only")) {
      numerics.hydro.av_qcap_center_band_only = strict_bool(
          hydro["av_qcap_center_band_only"],
          "Numerics.hydro.av_qcap_center_band_only");
    }
    const bool av_qcap_scope_explicit = has_key(hydro, "av_qcap_scope");
    if (av_qcap_scope_explicit) {
      numerics.hydro.av_qcap_scope = parse_av_qcap_scope(
          strict_string(hydro["av_qcap_scope"],
                        "Numerics.hydro.av_qcap_scope"),
          "Numerics.hydro.av_qcap_scope");
    } else if (numerics.hydro.av_qcap_center_band_only) {
      numerics.hydro.av_qcap_scope = AvQcapScope::TRI_FAN_RADIAL_INDEX;
    }
    if (av_qcap_scope_explicit && numerics.hydro.av_qcap_center_band_only &&
        numerics.hydro.av_qcap_scope != AvQcapScope::TRI_FAN_RADIAL_INDEX) {
      throw ConfigError(
          "conflicting q-cap scope and av_qcap_center_band_only");
    }
    if (has_key(hydro, "av_cfl_coefficient")) {
      numerics.hydro.av_cfl_coefficient = numeric_as_double(
          hydro["av_cfl_coefficient"],
          "Numerics.hydro.av_cfl_coefficient");
      ensure_positive(numerics.hydro.av_cfl_coefficient,
                      "Numerics.hydro.av_cfl_coefficient");
    }
    if (has_key(hydro, "csw_C1")) {
      numerics.hydro.csw_C1 =
          numeric_as_double(hydro["csw_C1"], "Numerics.hydro.csw_C1");
      ensure_non_negative(numerics.hydro.csw_C1, "Numerics.hydro.csw_C1");
    }
    if (has_key(hydro, "csw_C2")) {
      numerics.hydro.csw_C2 =
          numeric_as_double(hydro["csw_C2"], "Numerics.hydro.csw_C2");
      ensure_non_negative(numerics.hydro.csw_C2, "Numerics.hydro.csw_C2");
    }
    if (has_key(hydro, "csw_limiter")) {
      numerics.hydro.csw_limiter =
          strict_string(hydro["csw_limiter"], "Numerics.hydro.csw_limiter");
      if (!is_csw_limiter(numerics.hydro.csw_limiter)) {
        throw ValueError(
            "Numerics.hydro.csw_limiter must be one of {\"van_leer\", \"bj\"}");
      }
    }
    if (has_key(hydro, "csw_limiter_enabled")) {
      numerics.hydro.csw_limiter_enabled = strict_bool(
          hydro["csw_limiter_enabled"],
          "Numerics.hydro.csw_limiter_enabled");
    }
    if (has_key(hydro, "csw_axis_mirror_limiter")) {
      numerics.hydro.csw_axis_mirror_limiter = strict_bool(
          hydro["csw_axis_mirror_limiter"],
          "Numerics.hydro.csw_axis_mirror_limiter");
    }
    if (has_key(hydro, "csw_rz_lift_enabled")) {
      numerics.hydro.csw_rz_lift_enabled = strict_bool(
          hydro["csw_rz_lift_enabled"],
          "Numerics.hydro.csw_rz_lift_enabled");
    }
    if (has_key(hydro, "csw_rz_lift_guard_ratio")) {
      numerics.hydro.csw_rz_lift_guard_ratio = numeric_as_double(
          hydro["csw_rz_lift_guard_ratio"],
          "Numerics.hydro.csw_rz_lift_guard_ratio");
    }
    if (has_key(hydro, "csw_pole_floor_enabled")) {
      numerics.hydro.csw_pole_floor_enabled = strict_bool(
          hydro["csw_pole_floor_enabled"],
          "Numerics.hydro.csw_pole_floor_enabled");
    }
    if (has_key(hydro, "csw_pole_floor_sigma0")) {
      numerics.hydro.csw_pole_floor_sigma0 = numeric_as_double(
          hydro["csw_pole_floor_sigma0"],
          "Numerics.hydro.csw_pole_floor_sigma0");
    }
    if (has_key(hydro, "csw_pole_floor_theta0_rad")) {
      numerics.hydro.csw_pole_floor_theta0_rad = numeric_as_double(
          hydro["csw_pole_floor_theta0_rad"],
          "Numerics.hydro.csw_pole_floor_theta0_rad");
    }
    if (has_key(hydro, "csw_pole_floor_thetaf_rad")) {
      numerics.hydro.csw_pole_floor_thetaf_rad = numeric_as_double(
          hydro["csw_pole_floor_thetaf_rad"],
          "Numerics.hydro.csw_pole_floor_thetaf_rad");
    }
    if (has_key(hydro, "csw_pole_desens_enabled")) {
      numerics.hydro.csw_pole_desens_enabled = strict_bool(
          hydro["csw_pole_desens_enabled"],
          "Numerics.hydro.csw_pole_desens_enabled");
    }
    if (has_key(hydro, "csw_pole_desens_alpha")) {
      numerics.hydro.csw_pole_desens_alpha = numeric_as_double(
          hydro["csw_pole_desens_alpha"],
          "Numerics.hydro.csw_pole_desens_alpha");
    }
    if (has_key(hydro, "csw_pole_desens_theta0_rad")) {
      numerics.hydro.csw_pole_desens_theta0_rad = numeric_as_double(
          hydro["csw_pole_desens_theta0_rad"],
          "Numerics.hydro.csw_pole_desens_theta0_rad");
    }
    if (has_key(hydro, "csw_pole_desens_thetaf_rad")) {
      numerics.hydro.csw_pole_desens_thetaf_rad = numeric_as_double(
          hydro["csw_pole_desens_thetaf_rad"],
          "Numerics.hydro.csw_pole_desens_thetaf_rad");
    }
    if (has_key(hydro, "csw_polar_slaving_enabled")) {
      numerics.hydro.csw_polar_slaving_enabled = strict_bool(
          hydro["csw_polar_slaving_enabled"],
          "Numerics.hydro.csw_polar_slaving_enabled");
    }
    if (has_key(hydro, "csw_polar_slaving_min_columns")) {
      numerics.hydro.csw_polar_slaving_min_columns = strict_int32(
          hydro["csw_polar_slaving_min_columns"],
          "Numerics.hydro.csw_polar_slaving_min_columns");
    }
    if (has_key(hydro, "csw_polar_slaving_full_columns")) {
      numerics.hydro.csw_polar_slaving_full_columns = strict_int32(
          hydro["csw_polar_slaving_full_columns"],
          "Numerics.hydro.csw_polar_slaving_full_columns");
    }
    if (has_key(hydro, "csw_polar_slaving_outer_columns")) {
      numerics.hydro.csw_polar_slaving_outer_columns = strict_int32(
          hydro["csw_polar_slaving_outer_columns"],
          "Numerics.hydro.csw_polar_slaving_outer_columns");
    }
    if (has_key(hydro, "csw_polar_slaving_chi_on")) {
      numerics.hydro.csw_polar_slaving_chi_on = numeric_as_double(
          hydro["csw_polar_slaving_chi_on"],
          "Numerics.hydro.csw_polar_slaving_chi_on");
    }
    if (has_key(hydro, "csw_polar_slaving_chi_full")) {
      numerics.hydro.csw_polar_slaving_chi_full = numeric_as_double(
          hydro["csw_polar_slaving_chi_full"],
          "Numerics.hydro.csw_polar_slaving_chi_full");
    }
    if (has_key(hydro, "csw_polar_slaving_strength")) {
      numerics.hydro.csw_polar_slaving_strength = numeric_as_double(
          hydro["csw_polar_slaving_strength"],
          "Numerics.hydro.csw_polar_slaving_strength");
    }
    if (has_key(hydro, "csw_polar_slaving_av_stiffness_cfl_enabled")) {
      numerics.hydro.csw_polar_slaving_av_stiffness_cfl_enabled = strict_bool(
          hydro["csw_polar_slaving_av_stiffness_cfl_enabled"],
          "Numerics.hydro.csw_polar_slaving_av_stiffness_cfl_enabled");
    }
    if (has_key(hydro, "csw_polar_slaving_av_stiffness_sigma")) {
      numerics.hydro.csw_polar_slaving_av_stiffness_sigma = numeric_as_double(
          hydro["csw_polar_slaving_av_stiffness_sigma"],
          "Numerics.hydro.csw_polar_slaving_av_stiffness_sigma");
    }
    if (has_key(hydro, "wake_heat_flux_enabled")) {
      numerics.hydro.wake_heat_flux_enabled = strict_bool(
          hydro["wake_heat_flux_enabled"],
          "Numerics.hydro.wake_heat_flux_enabled");
    }
    if (has_key(hydro, "wake_heat_flux_CE")) {
      numerics.hydro.wake_heat_flux_CE = numeric_as_double(
          hydro["wake_heat_flux_CE"], "Numerics.hydro.wake_heat_flux_CE");
    }
    if (has_key(hydro, "wake_heat_flux_theta_a_rad")) {
      numerics.hydro.wake_heat_flux_theta_a_rad = numeric_as_double(
          hydro["wake_heat_flux_theta_a_rad"],
          "Numerics.hydro.wake_heat_flux_theta_a_rad");
    }
    if (has_key(hydro, "wake_heat_flux_theta_b_rad")) {
      numerics.hydro.wake_heat_flux_theta_b_rad = numeric_as_double(
          hydro["wake_heat_flux_theta_b_rad"],
          "Numerics.hydro.wake_heat_flux_theta_b_rad");
    }
    if (has_key(hydro, "wake_heat_flux_global_theta")) {
      numerics.hydro.wake_heat_flux_global_theta = strict_bool(
          hydro["wake_heat_flux_global_theta"],
          "Numerics.hydro.wake_heat_flux_global_theta");
    }
    if (has_key(hydro, "csw_shock_limiter_floor")) {
      numerics.hydro.csw_shock_limiter_floor = numeric_as_double(
          hydro["csw_shock_limiter_floor"],
          "Numerics.hydro.csw_shock_limiter_floor");
      if (!(numerics.hydro.csw_shock_limiter_floor >= 0.0 &&
            numerics.hydro.csw_shock_limiter_floor <= 1.0)) {
        throw ValueError("Numerics.hydro.csw_shock_limiter_floor must be in [0, 1]");
      }
    }
    if (has_key(hydro, "csw_zero_uniform_compression")) {
      numerics.hydro.csw_zero_uniform_compression = strict_bool(
          hydro["csw_zero_uniform_compression"],
          "Numerics.hydro.csw_zero_uniform_compression");
    }
    if (has_key(hydro, "csw_diagnostics")) {
      numerics.hydro.csw_diagnostics =
          strict_bool(hydro["csw_diagnostics"], "Numerics.hydro.csw_diagnostics");
    }
    if (has_key(hydro, "av_limiter_J")) {
      numerics.hydro.av_limiter_J =
          numeric_as_double(hydro["av_limiter_J"], "Numerics.hydro.av_limiter_J");
      ensure_non_negative(numerics.hydro.av_limiter_J, "Numerics.hydro.av_limiter_J");
    }
    if (has_key(hydro, "av_heat_C")) {
      numerics.hydro.av_heat_C =
          numeric_as_double(hydro["av_heat_C"], "Numerics.hydro.av_heat_C");
      ensure_non_negative(numerics.hydro.av_heat_C, "Numerics.hydro.av_heat_C");
    }
    if (has_key(hydro, "post_shock_heat")) {
      numerics.hydro.post_shock_heat =
          strict_bool(hydro["post_shock_heat"], "Numerics.hydro.post_shock_heat");
    }
    if (has_key(hydro, "post_shock_heat_C")) {
      numerics.hydro.post_shock_heat_C =
          numeric_as_double(hydro["post_shock_heat_C"],
                            "Numerics.hydro.post_shock_heat_C");
      ensure_non_negative(numerics.hydro.post_shock_heat_C,
                          "Numerics.hydro.post_shock_heat_C");
    }
    if (has_key(hydro, "post_shock_heat_decay")) {
      numerics.hydro.post_shock_heat_decay =
          numeric_as_double(hydro["post_shock_heat_decay"],
                            "Numerics.hydro.post_shock_heat_decay");
      ensure_positive(numerics.hydro.post_shock_heat_decay,
                      "Numerics.hydro.post_shock_heat_decay");
    }
    if (has_key(hydro, "post_shock_velocity_damping_C")) {
      numerics.hydro.post_shock_velocity_damping_C =
          numeric_as_double(hydro["post_shock_velocity_damping_C"],
                            "Numerics.hydro.post_shock_velocity_damping_C");
      ensure_non_negative(numerics.hydro.post_shock_velocity_damping_C,
                          "Numerics.hydro.post_shock_velocity_damping_C");
    }
    if (has_key(hydro, "bulk_viscosity_C")) {
      numerics.hydro.bulk_viscosity_C =
          numeric_as_double(hydro["bulk_viscosity_C"],
                            "Numerics.hydro.bulk_viscosity_C");
      ensure_non_negative(numerics.hydro.bulk_viscosity_C,
                          "Numerics.hydro.bulk_viscosity_C");
    }
    if (has_key(hydro, "ion_art_heat_C")) {
      numerics.hydro.ion_art_heat_C =
          numeric_as_double(hydro["ion_art_heat_C"],
                            "Numerics.hydro.ion_art_heat_C");
      ensure_non_negative(numerics.hydro.ion_art_heat_C,
                          "Numerics.hydro.ion_art_heat_C");
    }
    if (has_key(hydro, "crossing_dt_safety")) {
      numerics.hydro.crossing_dt_safety =
          numeric_as_double(hydro["crossing_dt_safety"],
                            "Numerics.hydro.crossing_dt_safety");
      ensure_non_negative(numerics.hydro.crossing_dt_safety,
                          "Numerics.hydro.crossing_dt_safety");
    }
    if (has_key(hydro, "time_integrator")) {
      numerics.hydro.time_integrator =
          strict_string(hydro["time_integrator"],
                        "Numerics.hydro.time_integrator");
      if (numerics.hydro.time_integrator != "legacy_pc" &&
          numerics.hydro.time_integrator != "midpoint_v2") {
        throw ValueError(
            "Numerics.hydro.time_integrator must be \"legacy_pc\" or "
            "\"midpoint_v2\"");
      }
    }
    if (has_key(hydro, "adaptive_av")) {
      const py::handle adaptive_obj = hydro["adaptive_av"];
      if (!py::isinstance<py::dict>(adaptive_obj)) {
        throw_value_type_error("Numerics.hydro.adaptive_av", "dict", adaptive_obj);
      }
      const py::dict adaptive = py::reinterpret_borrow<py::dict>(adaptive_obj);
      enforce_known_keys(adaptive, "Numerics.hydro.adaptive_av",
                         {"enabled", "base", "primary", "rebound",
                          "taper_r_start", "taper_r_end", "hysteresis_w",
                          "hysteresis_tau",
                          "support_ahead", "support_behind"});
      auto& av_adapt = numerics.hydro.adaptive_av;
      if (has_key(adaptive, "enabled")) {
        av_adapt.enabled =
            strict_bool(adaptive["enabled"], "Numerics.hydro.adaptive_av.enabled");
      }
      if (has_key(adaptive, "base")) {
        parse_adaptive_av_coeff(adaptive["base"], av_adapt.base,
                                "Numerics.hydro.adaptive_av.base");
      }
      if (has_key(adaptive, "primary")) {
        parse_adaptive_av_coeff(adaptive["primary"], av_adapt.primary,
                                "Numerics.hydro.adaptive_av.primary");
      }
      if (has_key(adaptive, "rebound")) {
        parse_adaptive_av_coeff(adaptive["rebound"], av_adapt.rebound,
                                "Numerics.hydro.adaptive_av.rebound");
      }
      if (has_key(adaptive, "taper_r_start")) {
        av_adapt.taper_r_start = numeric_as_double(
            adaptive["taper_r_start"], "Numerics.hydro.adaptive_av.taper_r_start");
      }
      if (has_key(adaptive, "taper_r_end")) {
        av_adapt.taper_r_end = numeric_as_double(
            adaptive["taper_r_end"], "Numerics.hydro.adaptive_av.taper_r_end");
      }
      if (has_key(adaptive, "hysteresis_w")) {
        av_adapt.hysteresis_w = numeric_as_double(
            adaptive["hysteresis_w"], "Numerics.hydro.adaptive_av.hysteresis_w");
      }
      if (has_key(adaptive, "hysteresis_tau")) {
        av_adapt.hysteresis_tau = numeric_as_double(
            adaptive["hysteresis_tau"],
            "Numerics.hydro.adaptive_av.hysteresis_tau");
        ensure_non_negative(av_adapt.hysteresis_tau,
                            "Numerics.hydro.adaptive_av.hysteresis_tau");
      }
      if (has_key(adaptive, "support_ahead")) {
        av_adapt.support_ahead = strict_int32(
            adaptive["support_ahead"], "Numerics.hydro.adaptive_av.support_ahead");
      }
      if (has_key(adaptive, "support_behind")) {
        av_adapt.support_behind = strict_int32(
            adaptive["support_behind"], "Numerics.hydro.adaptive_av.support_behind");
      }
      ensure_non_negative(av_adapt.taper_r_start,
                          "Numerics.hydro.adaptive_av.taper_r_start");
      ensure_non_negative(av_adapt.taper_r_end,
                          "Numerics.hydro.adaptive_av.taper_r_end");
      if (!(av_adapt.taper_r_start > av_adapt.taper_r_end)) {
        throw ValueError(
            "Numerics.hydro.adaptive_av.taper_r_start must be > taper_r_end");
      }
      if (!(av_adapt.hysteresis_w >= 0.0 && av_adapt.hysteresis_w <= 1.0)) {
        throw ValueError("Numerics.hydro.adaptive_av.hysteresis_w must be in [0, 1]");
      }
      ensure_int_ge(av_adapt.support_ahead, 0,
                    "Numerics.hydro.adaptive_av.support_ahead");
      ensure_int_ge(av_adapt.support_behind, 0,
                    "Numerics.hydro.adaptive_av.support_behind");
    }
    if (has_key(hydro, "av_eos_aware")) {
      numerics.hydro.av_eos_aware =
          strict_bool(hydro["av_eos_aware"], "Numerics.hydro.av_eos_aware");
    }
    if (has_key(hydro, "av_eos_gamma1_ref")) {
      numerics.hydro.av_eos_gamma1_ref =
          numeric_as_double(hydro["av_eos_gamma1_ref"],
                            "Numerics.hydro.av_eos_gamma1_ref");
      ensure_positive(numerics.hydro.av_eos_gamma1_ref,
                      "Numerics.hydro.av_eos_gamma1_ref");
    }
    if (has_key(hydro, "av_eos_boost_max")) {
      numerics.hydro.av_eos_boost_max =
          numeric_as_double(hydro["av_eos_boost_max"],
                            "Numerics.hydro.av_eos_boost_max");
      ensure_positive(numerics.hydro.av_eos_boost_max,
                      "Numerics.hydro.av_eos_boost_max");
      if (numerics.hydro.av_eos_boost_max < 1.0) {
        throw ConfigError("Numerics.hydro.av_eos_boost_max must be >= 1");
      }
    }
    if (has_key(hydro, "odd_even_damping_C")) {
      numerics.hydro.odd_even_damping_C =
          numeric_as_double(hydro["odd_even_damping_C"],
                            "Numerics.hydro.odd_even_damping_C");
      ensure_non_negative(numerics.hydro.odd_even_damping_C,
                          "Numerics.hydro.odd_even_damping_C");
    }
    if (has_key(hydro, "ee_odd_even_C")) {
      numerics.hydro.ee_odd_even_C =
          numeric_as_double(hydro["ee_odd_even_C"],
                            "Numerics.hydro.ee_odd_even_C");
      ensure_non_negative(numerics.hydro.ee_odd_even_C,
                          "Numerics.hydro.ee_odd_even_C");
    }
    if (has_key(hydro, "hk_velocity_damper_C")) {
      numerics.hydro.hk_velocity_damper_C =
          numeric_as_double(hydro["hk_velocity_damper_C"],
                            "Numerics.hydro.hk_velocity_damper_C");
      ensure_non_negative(numerics.hydro.hk_velocity_damper_C,
                          "Numerics.hydro.hk_velocity_damper_C");
    }
    if (has_key(hydro, "hk_velocity_damper_tau_min")) {
      numerics.hydro.hk_velocity_damper_tau_min =
          numeric_as_double(hydro["hk_velocity_damper_tau_min"],
                            "Numerics.hydro.hk_velocity_damper_tau_min");
      ensure_non_negative(numerics.hydro.hk_velocity_damper_tau_min,
                          "Numerics.hydro.hk_velocity_damper_tau_min");
    }
    if (has_key(hydro, "hk_velocity_damper_grad_Te_max")) {
      numerics.hydro.hk_velocity_damper_grad_Te_max =
          numeric_as_double(hydro["hk_velocity_damper_grad_Te_max"],
                            "Numerics.hydro.hk_velocity_damper_grad_Te_max");
      ensure_non_negative(numerics.hydro.hk_velocity_damper_grad_Te_max,
                          "Numerics.hydro.hk_velocity_damper_grad_Te_max");
    }
    if (has_key(hydro, "hk_velocity_damper_grad_rho_max")) {
      numerics.hydro.hk_velocity_damper_grad_rho_max =
          numeric_as_double(hydro["hk_velocity_damper_grad_rho_max"],
                            "Numerics.hydro.hk_velocity_damper_grad_rho_max");
      ensure_non_negative(numerics.hydro.hk_velocity_damper_grad_rho_max,
                          "Numerics.hydro.hk_velocity_damper_grad_rho_max");
    }
    if (has_key(hydro, "hk_velocity_damper_guard_cells")) {
      numerics.hydro.hk_velocity_damper_guard_cells =
          strict_int32(hydro["hk_velocity_damper_guard_cells"],
                       "Numerics.hydro.hk_velocity_damper_guard_cells");
      ensure_int_ge(numerics.hydro.hk_velocity_damper_guard_cells, 0,
                    "Numerics.hydro.hk_velocity_damper_guard_cells");
    }
    if (has_key(hydro, "av_heat_to")) {
      numerics.hydro.av_heat_to =
          strict_string(hydro["av_heat_to"], "Numerics.hydro.av_heat_to");
    }
    if (has_key(hydro, "pressure_drive_perturbation")) {
      const py::handle perturbation_obj =
          hydro["pressure_drive_perturbation"];
      if (!py::isinstance<py::dict>(perturbation_obj)) {
        throw_value_type_error(
            "Numerics.hydro.pressure_drive_perturbation",
            "dict",
            perturbation_obj);
      }
      const py::dict perturbation =
          py::reinterpret_borrow<py::dict>(perturbation_obj);
      enforce_known_keys(perturbation,
                         "Numerics.hydro.pressure_drive_perturbation",
                         {"enabled",
                          "legendre_modes",
                          "ring_spots",
                          "random_seed",
                          "random_l_min",
                          "random_l_max",
                          "random_rms"});
      auto& perturbation_config =
          numerics.hydro.pressure_drive_perturbation;
      if (has_key(perturbation, "enabled")) {
        perturbation_config.enabled = strict_bool(
            perturbation["enabled"],
            "Numerics.hydro.pressure_drive_perturbation.enabled");
      }
      if (has_key(perturbation, "legendre_modes")) {
        const py::handle modes_obj = perturbation["legendre_modes"];
        if (!py::isinstance<py::sequence>(modes_obj) ||
            py::isinstance<py::str>(modes_obj)) {
          throw ConfigError(
              "Numerics.hydro.pressure_drive_perturbation.legendre_modes "
              "must be a list of [l, amplitude] pairs");
        }
        const py::sequence modes =
            py::reinterpret_borrow<py::sequence>(modes_obj);
        if (modes.size() >
            tenryu::hydro::PressureDrivePerturbationParams::kMaxModes) {
          throw ConfigError(
              "Numerics.hydro.pressure_drive_perturbation has more than 24 "
              "resolved modes");
        }
        for (std::size_t k = 0; k < modes.size(); ++k) {
          const py::handle mode_obj = modes[k];
          if (!py::isinstance<py::sequence>(mode_obj) ||
              py::isinstance<py::str>(mode_obj)) {
            throw ConfigError(
                "Numerics.hydro.pressure_drive_perturbation.legendre_modes "
                "entries must be [l, amplitude] pairs");
          }
          const py::sequence mode =
              py::reinterpret_borrow<py::sequence>(mode_obj);
          if (mode.size() != 2) {
            throw ConfigError(
                "Numerics.hydro.pressure_drive_perturbation.legendre_modes "
                "entries must contain exactly two values");
          }
          const std::string base =
              "Numerics.hydro.pressure_drive_perturbation.legendre_modes[" +
              std::to_string(k) + "]";
          const int l = strict_int32(mode[0], base + "[0]");
          if (l < 1 || l > 16) {
            throw ConfigError(base + "[0] must be in [1, 16]");
          }
          perturbation_config.mode_l.push_back(l);
          perturbation_config.mode_a.push_back(
              pressure_drive_amplitude_as_double(mode[1], base + "[1]"));
        }
      }
      if (has_key(perturbation, "ring_spots")) {
        const py::handle spots_obj = perturbation["ring_spots"];
        if (!py::isinstance<py::sequence>(spots_obj) ||
            py::isinstance<py::str>(spots_obj)) {
          throw ConfigError(
              "Numerics.hydro.pressure_drive_perturbation.ring_spots must "
              "be a list of [theta0, sigma, amplitude] triples");
        }
        const py::sequence spots =
            py::reinterpret_borrow<py::sequence>(spots_obj);
        if (spots.size() >
            tenryu::hydro::PressureDrivePerturbationParams::kMaxSpots) {
          throw ConfigError(
              "Numerics.hydro.pressure_drive_perturbation has more than 4 "
              "ring spots");
        }
        for (std::size_t m = 0; m < spots.size(); ++m) {
          const py::handle spot_obj = spots[m];
          if (!py::isinstance<py::sequence>(spot_obj) ||
              py::isinstance<py::str>(spot_obj)) {
            throw ConfigError(
                "Numerics.hydro.pressure_drive_perturbation.ring_spots "
                "entries must be [theta0, sigma, amplitude] triples");
          }
          const py::sequence spot =
              py::reinterpret_borrow<py::sequence>(spot_obj);
          if (spot.size() != 3) {
            throw ConfigError(
                "Numerics.hydro.pressure_drive_perturbation.ring_spots "
                "entries must contain exactly three values");
          }
          const std::string base =
              "Numerics.hydro.pressure_drive_perturbation.ring_spots[" +
              std::to_string(m) + "]";
          const double theta0 = numeric_as_double(spot[0], base + "[0]");
          const double sigma = numeric_as_double(spot[1], base + "[1]");
          const double amplitude =
              pressure_drive_amplitude_as_double(spot[2], base + "[2]");
          if (theta0 < 0.0 || theta0 > M_PI) {
            throw ConfigError(base + "[0] must be in [0, pi]");
          }
          if (!(sigma > 0.0)) {
            throw ConfigError(base + "[1] must be > 0");
          }
          perturbation_config.spot_theta0.push_back(theta0);
          perturbation_config.spot_sigma.push_back(sigma);
          perturbation_config.spot_amp.push_back(amplitude);
        }
      }
      if (has_key(perturbation, "random_seed")) {
        perturbation_config.random_seed = static_cast<long long>(strict_int64(
            perturbation["random_seed"],
            "Numerics.hydro.pressure_drive_perturbation.random_seed"));
      }
      if (has_key(perturbation, "random_l_min")) {
        perturbation_config.random_l_min = strict_int32(
            perturbation["random_l_min"],
            "Numerics.hydro.pressure_drive_perturbation.random_l_min");
      }
      if (has_key(perturbation, "random_l_max")) {
        perturbation_config.random_l_max = strict_int32(
            perturbation["random_l_max"],
            "Numerics.hydro.pressure_drive_perturbation.random_l_max");
      }
      if (has_key(perturbation, "random_rms")) {
        perturbation_config.random_rms = pressure_drive_amplitude_as_double(
            perturbation["random_rms"],
            "Numerics.hydro.pressure_drive_perturbation.random_rms");
      }
      if (perturbation_config.random_l_min < 1 ||
          perturbation_config.random_l_min >
              perturbation_config.random_l_max ||
          perturbation_config.random_l_max > 16) {
        throw ConfigError(
            "Numerics.hydro.pressure_drive_perturbation random l range must "
            "satisfy 1 <= random_l_min <= random_l_max <= 16");
      }
      resolve_random_pressure_drive_modes(perturbation_config);
      if (perturbation_config.mode_l.size() >
          static_cast<std::size_t>(
              tenryu::hydro::PressureDrivePerturbationParams::kMaxModes)) {
        throw ConfigError(
            "Numerics.hydro.pressure_drive_perturbation has more than 24 "
            "resolved modes");
      }
      if (perturbation_config.enabled &&
          perturbation_config.mode_l.empty() &&
          perturbation_config.spot_theta0.empty()) {
        throw ConfigError("perturbation enabled but empty");
      }
      compute_pressure_drive_perturbation_range(perturbation_config);
    }
    if (has_key(hydro, "boundary_pressure")) {
      const auto callable = extract_callable_or_throw(hydro["boundary_pressure"],
                                                      "Numerics.hydro.boundary_pressure");
      numerics.hydro.pressure_drive_1d = to_config_callable(callable);
      register_callable("Numerics.hydro.boundary_pressure", callable,
                        hydro["boundary_pressure"]);
    }
  }

  if (has_key(kwargs, "conduction")) {
    const py::handle conduction_obj = kwargs["conduction"];
    if (!py::isinstance<py::dict>(conduction_obj)) {
      throw_value_type_error("Numerics.conduction", "dict", conduction_obj);
    }
    const py::dict conduction = py::reinterpret_borrow<py::dict>(conduction_obj);
    enforce_known_keys(conduction, "Numerics.conduction",
                       {"enabled", "allow_single_material_approximation",
                        "solver", "sts_floor_limiter", "ion_conduction", "f_lim",
                        "mfp_limiter_C", "spitzer_z_correction", "sts_damping",
                        "sts_max_stages", "sts_subcycle_eta",
                        "sts_total_stages_max", "halo_strategy", "hypre_rtol",
                        "hypre_max_iter", "test_kappa", "test_planar",
                        "face_kappa_policy",
                        "nonlocal_model", "snb_n_groups", "snb_E_max_over_Te", "snb_mfp",
                        "snb_efield", "snb_picard_max_iters", "snb_picard_rtol"});
    if (has_key(conduction, "allow_single_material_approximation")) {
      warn_ignored_key("Numerics.conduction.allow_single_material_approximation");
    }
    if (has_key(conduction, "enabled")) {
      numerics.conduction.enabled =
          strict_bool(conduction["enabled"], "Numerics.conduction.enabled");
    }
    if (has_key(conduction, "solver")) {
      numerics.conduction.solver =
          strict_string(conduction["solver"], "Numerics.conduction.solver");
    }
    if (has_key(conduction, "sts_floor_limiter")) {
      numerics.conduction.sts_floor_limiter = strict_string(
          conduction["sts_floor_limiter"], "Numerics.conduction.sts_floor_limiter");
    }
    if (has_key(conduction, "ion_conduction")) {
      numerics.conduction.ion_conduction =
          strict_bool(conduction["ion_conduction"], "Numerics.conduction.ion_conduction");
    }
    if (has_key(conduction, "f_lim")) {
      numerics.conduction.f_lim =
          numeric_as_double(conduction["f_lim"], "Numerics.conduction.f_lim");
    }
    if (has_key(conduction, "mfp_limiter_C")) {
      numerics.conduction.mfp_limiter_C = numeric_as_double(
          conduction["mfp_limiter_C"], "Numerics.conduction.mfp_limiter_C");
    }
    if (has_key(conduction, "spitzer_z_correction")) {
      const std::string spitzer_z_correction = strict_string(
          conduction["spitzer_z_correction"],
          "Numerics.conduction.spitzer_z_correction");
      if (spitzer_z_correction == "off") {
        throw ConfigError(
            "the fixed-coefficient Spitzer path was removed 2026-07-30; "
            "gamma0(Z) is always applied");
      }
      if (spitzer_z_correction != "auto" &&
          spitzer_z_correction != "epperlein_short") {
        throw ConfigError(
            "Numerics.conduction.spitzer_z_correction must be \"auto\", "
            "\"off\", or \"epperlein_short\"");
      }
      tenryu::core::log_info(
          "spitzer_z_correction is always on "
          "(gamma0(Z) folded into the base formula 2026-07-30)");
    }
    if (has_key(conduction, "sts_damping")) {
      numerics.conduction.sts_damping = numeric_as_double(
          conduction["sts_damping"], "Numerics.conduction.sts_damping");
    }
    if (has_key(conduction, "sts_max_stages")) {
      numerics.conduction.sts_max_stages = strict_int32(
          conduction["sts_max_stages"], "Numerics.conduction.sts_max_stages");
    }
    if (has_key(conduction, "sts_subcycle_eta")) {
      numerics.conduction.sts_subcycle_eta = numeric_as_double(
          conduction["sts_subcycle_eta"], "Numerics.conduction.sts_subcycle_eta");
    }
    if (has_key(conduction, "sts_total_stages_max")) {
      numerics.conduction.sts_total_stages_max = strict_int32(
          conduction["sts_total_stages_max"],
          "Numerics.conduction.sts_total_stages_max");
    }
    if (has_key(conduction, "halo_strategy")) {
      numerics.conduction.halo_strategy = strict_string(
          conduction["halo_strategy"], "Numerics.conduction.halo_strategy");
    }
    if (has_key(conduction, "hypre_rtol")) {
      numerics.conduction.hypre_rtol = numeric_as_double(
          conduction["hypre_rtol"], "Numerics.conduction.hypre_rtol");
    }
    if (has_key(conduction, "hypre_max_iter")) {
      numerics.conduction.hypre_max_iter = strict_int32(
          conduction["hypre_max_iter"], "Numerics.conduction.hypre_max_iter");
    }
    if (has_key(conduction, "test_kappa")) {
      numerics.conduction.test_kappa =
          numeric_as_double(conduction["test_kappa"], "Numerics.conduction.test_kappa");
    }
    if (has_key(conduction, "test_planar")) {
      numerics.conduction.test_planar =
          strict_bool(conduction["test_planar"], "Numerics.conduction.test_planar");
    }
    if (has_key(conduction, "face_kappa_policy")) {
      numerics.conduction.face_kappa_policy = strict_string(
          conduction["face_kappa_policy"], "Numerics.conduction.face_kappa_policy");
    }
    if (has_key(conduction, "nonlocal_model")) {
      numerics.conduction.nonlocal_model =
          strict_string(conduction["nonlocal_model"], "Numerics.conduction.nonlocal_model");
    }
    if (has_key(conduction, "snb_n_groups")) {
      numerics.conduction.snb_n_groups = strict_int32(
          conduction["snb_n_groups"], "Numerics.conduction.snb_n_groups");
    }
    if (has_key(conduction, "snb_E_max_over_Te")) {
      numerics.conduction.snb_E_max_over_Te = numeric_as_double(
          conduction["snb_E_max_over_Te"], "Numerics.conduction.snb_E_max_over_Te");
    }
    if (has_key(conduction, "snb_mfp")) {
      numerics.conduction.snb_mfp =
          strict_string(conduction["snb_mfp"], "Numerics.conduction.snb_mfp");
    }
    if (has_key(conduction, "snb_efield")) {
      numerics.conduction.snb_efield =
          strict_string(conduction["snb_efield"], "Numerics.conduction.snb_efield");
    }
    if (has_key(conduction, "snb_picard_max_iters")) {
      numerics.conduction.snb_picard_max_iters = strict_int32(
          conduction["snb_picard_max_iters"], "Numerics.conduction.snb_picard_max_iters");
    }
    if (has_key(conduction, "snb_picard_rtol")) {
      numerics.conduction.snb_picard_rtol = numeric_as_double(
          conduction["snb_picard_rtol"], "Numerics.conduction.snb_picard_rtol");
    }
  }

  if (has_key(kwargs, "ale")) {
    const py::handle ale_obj = kwargs["ale"];
    if (!py::isinstance<py::dict>(ale_obj)) {
      throw_value_type_error("Numerics.ale", "dict", ale_obj);
    }
    const py::dict ale = py::reinterpret_borrow<py::dict>(ale_obj);
    enforce_known_keys(ale, "Numerics.ale",
                       {"enabled", "mesh_mode", "reale_core",
                        "rezone_min_dt_s",
                        "tess_gpu_dual",
                        "tess_gpu_restrict",
                        "dvclp_solver_rev",
                        "reale_lloyd_max",
                        "reale_short_edge_collapse_rel",
                        "reale_subdomain_rezone",
                        "reale_subdomain_frac_max",
                        "reale_overlay_additivity_tol",
                        "reale_corner_mass_reset",
                        "reale_velocity_max_principle",
                        "reale_dt_trigger_factor",
                        "reale_dt_trigger_cooldown",
                        "ale_identity_mode",
                        "ale_mover_diag",
                        "ale_preserve_lagrangian_velocity_carry",
                        "align_diagnostics",
                        "every_n_steps", "warmup_steps",
                        "relaxation", "spacing_ratio_threshold", "quality_threshold",
                        "max_iterations", "convergence_tol",
                        "max_displacement_fraction", "remap_limiter",
                        "swept_volume_sign_fixed", "donor_sign_fixed",
                        "remap_ms_midpoint", "remap_ms_post_check",
                        "remap_ms_post_max_iter", "remap_ms_rescale_floor",
                        "ke_fixup", "ke_conservation_closure",
                        "ke_conservation_closure_audit",
                        "ke_closure_redistribute_floor",
                        "debug_per_remap_log",
                        "shock_sensor_guard_cells", "density_jump_threshold",
                        "Te_jump_threshold", "preventive_axis_guard_fraction",
                        "axis_z_motion", "winslow_axis_kappa",
                        "button_morph",
                        "runtime_controller",
                        "reference_barrier_enabled",
                        "reference_target",
                        "reference_blend_default",
                        "reference_volume_floor_rel",
                        "reference_corner_j_floor_rel",
                        "reference_gauss_j_floor_rel",
                        "reference_linesearch_max_iters",
                        "reference_force_engage_every_step",
                        "reference_trigger_axis_margin_enabled",
                        "reference_trigger_axis_margin_threshold",
                        "reference_trigger_corner_j_ratio_enabled",
                        "reference_trigger_corner_j_ratio_threshold",
                        "dgcl_commit_gate",
                        "transaction_failure_inject_point",
                        "dgcl_commit_rtol",
                        "driver_retry_reference_barrier_enabled",
                        "driver_retry_reference_barrier_K_axis",
                        "driver_retry_reference_barrier_eta_axis",
                        "driver_retry_reference_barrier_max_attempts",
                        "driver_retry_reference_barrier_same_sig_max",
                        "driver_retry_reference_barrier_cell_window",
                        "driver_retry_reference_barrier_dt_collapse_rel",
                        "driver_retry_reference_barrier_lambda_collapse_threshold",
                        "driver_retry_reference_barrier_lambda_collapse_count",
                        "driver_retry_reference_barrier_quality_progress_factor",
                        "driver_retry_reference_barrier_quality_progress_count",
                        "driver_retry_reference_barrier_rezone_freq_warn_fraction",
                        "driver_retry_reference_barrier_rezone_freq_window",
                        "driver_retry_reference_barrier_chi",
                        "driver_retry_reference_barrier_q_retry",
                        "remap_damage_gate_enabled",
                        "remap_damage_dmax", "remap_damage_axis_eta",
                          "remap_damage_axis_budget_enabled",
                          "remap_damage_axis_budget_factor",
                          "predictive_acceptance_enabled",
                          "predictive_acceptance_axis_floor_fraction",
                          "predictive_acceptance_cell_vol_floor_fraction",
                          "safe_backtrack_enabled",
                          "safe_backtrack_min_exp",
                          "safe_backtrack_binary_iters",
                          "mesh_epoch_enabled",
                          "mesh_epoch_max_per_step",
                          "corner_cell_aspect_protection_enabled",
                          "corner_cell_aspect_eta",
                          "rezone_solver",
                          "m1_gamma_align",
                          "m1_lambda_tether",
                          "m1_theta_reg",
                          "m1_sweeps",
                          "m1_min_j_dec_rel",
                          "m1_barrier_beta",
                          "euler_window",
                          "euler_windows",
                          "band_ale",
                          "evacuated_cell",
                          "rezone_local_admissibility_linesearch",
                        "rezone_local_j_floor_rel",
                        "rezone_local_linesearch_max_halves",
                        "reject_zero_gauss_j",
                        "zero_gauss_j_floor_rel",
                        "lambda_sweep_diagnostic_enabled",
                        "lambda_sweep_target_cell_c",
                        "lambda_sweep_target_cell_i",
                        "lambda_sweep_target_cell_j",
                        "lambda_sweep_max_exp",
                        "corner_jacobian_post_tangle_enabled",
                        "corner_post_tangle_strict_floor_enabled",
                        "local_boundary_repair_enabled",
                        "multi_node_boundary_repair_enabled",
                        "multi_node_interior_repair_enabled",
                        "axis_variational_projection_enabled",
                        "emergency_cell_deactivation_enabled",
                        "multiblock_cross_seam_rezone_enabled",
                        "multiblock_scaled_reference_enabled",
                        "multiblock_differential_reference_enabled",
                        "multiblock_differential_reference_band_count",
                        "multiblock_differential_reference_smoothing_g0",
                        "multiblock_differential_reference_nu",
                        "multiblock_differential_reference_eps_v",
                        "multiblock_differential_reference_s_cap_min_rel",
                        "multiblock_differential_reference_xi_seam_tol",
                        "multiblock_differential_reference_sigma_warn_floor",
                        "multiblock_lagrangian_bulk_center_patch_reference_enabled",
                        "multiblock_center_patch_ring_max",
                        "multiblock_center_patch_xi_center",
                        "multiblock_center_patch_halo_layers",
                        "multiblock_center_patch_vol_on",
                        "multiblock_center_patch_vol_off",
                        "multiblock_center_patch_cornerj_on",
                        "multiblock_center_patch_cornerj_off",
                        "multiblock_center_patch_gaussj_on",
                        "multiblock_center_patch_gaussj_off",
                        "ale_reference_diagnostics_enabled",
                        "multiblock_path_admissibility_enabled",
                        "path_admissibility_floor",
                        "dt_rejection_factor",
                        "max_dt_rejections",
                        "axis_band_managed_remap_enabled",
                        "axis_band_managed_remap_width",
                        "axis_band_managed_remap_max_width",
                        "axis_band_managed_remap_every_hydro_half_step",
                        "axis_band_managed_remap_margin_trigger",
                        "axis_band_managed_remap_equal_volume",
                        "axis_band_managed_remap_include_radiation_groups",
                        "axis_rezone_enabled",
                        "axis_rezone_trigger_edge_fraction",
                        "axis_rezone_trigger_min_altitude_fraction",
                        "axis_rezone_eta_floor",
                        "core_freeze_enabled",
                        "core_freeze_source",
                        "core_freeze_tracer_cut",
                        "core_freeze_halo_layers",
                        "core_freeze_apply_to_axis_rezone",
                        "core_freeze_skip_velocity_projection",
                        "axis_repair_mode",
                        "remap_scheme", "remap_ms2_limiter",
                        "conservative_remap_enabled",
                        "conservative_remap_target",
                        "conservative_remap_radiation_enabled",
                        "conservative_remap_order",
                        "tri_fan_tracking_reference_enabled",
                        "tri_fan_tracking_reference_mode",
                        "tri_fan_tracking_reference_omega",
                        "tri_fan_tracking_reference_beta",
                        "tri_fan_tracking_reference_g0",
                        "tri_fan_tracking_reference_nu",
                        "tri_fan_tracking_reference_eps_v",
                        "conservative_remap_lagrangian_bulk_enabled",
                        "conservative_remap_lagrangian_bulk_center_node_ring_max",
                        "central_pseudo_core_enabled",
                        "central_pseudo_core_s_c",
                        "central_pseudo_core_activation_time_s",
                        "central_pseudo_core_ring_absorption_enabled",
                        "central_pseudo_core_ring_absorption_tau",
                        "conv_rezone_enabled",
                        "central_pseudo_core_core1d_enabled",
                        "central_pseudo_core_core1d_build_shells",
                        "central_pseudo_core_core1d_split_append",
                        "central_pseudo_core_core1d_av_c1",
                        "central_pseudo_core_core1d_av_c2",
                        "central_pseudo_core_core1d_cfl",
                        "central_pseudo_core_core1d_piston_cap",
                        "central_pseudo_core_core1d_max_substeps",
                        "central_pseudo_core_core1d_dist_append",
                        "central_pseudo_core_spherical_absorb_gasfront",
                        "central_pseudo_core_spherical_absorb_alpha",
                        "central_pseudo_core_spherical_absorb_pjump",
                        "central_pseudo_core_mixed_absorb_enabled",
                        "central_pseudo_core_absorb_watch_rows",
                        "remap_mass_closure_reject_tol",
                        "rezone_closure_cooldown_steps",
                        "csr_optionb_coherent_enabled",
                        "csr_optionb_velocity_remap_enabled",
                        "pole_axis_bbsw_enabled",
                        "axis_contact_guard_enabled",
                        "mass_floor_absorb_enabled",
                        "interior_patch_remap_enabled",
                        "central_pseudo_core_ring_absorption_max_rings",
                        "central_pseudo_core_ring_absorption_gas_tracer_min",
                        "central_pseudo_core_ring_absorption_gas_tracer_cell_min",
                        "pole_sector_rezone_enabled",
                        "pole_sector_rezone_m_theta",
                        "pole_sector_rezone_lambda",
                        "pole_sector_rezone_mode",
                        "pole_sector_rezone_deadband_frac",
                        "force_rezone_every_n_steps"});
    const bool remap_limiter_explicit = has_key(ale, "remap_limiter");
    const bool remap_ms_midpoint_explicit = has_key(ale, "remap_ms_midpoint");
    const bool remap_ms_post_check_explicit = has_key(ale, "remap_ms_post_check");
    if (has_key(ale, "enabled")) {
      numerics.ale.enabled = strict_bool(ale["enabled"], "Numerics.ale.enabled");
    }
    if (has_key(ale, "mesh_mode")) {
      numerics.ale.mesh_mode =
          strict_string(ale["mesh_mode"], "Numerics.ale.mesh_mode");
    }
    if (has_key(ale, "reale_core")) {
      numerics.ale.reale_core =
          strict_string(ale["reale_core"], "Numerics.ale.reale_core");
    }
    if (has_key(ale, "rezone_min_dt_s")) {
      numerics.ale.rezone_min_dt_s = numeric_as_double(
          ale["rezone_min_dt_s"], "Numerics.ale.rezone_min_dt_s");
    }
    if (has_key(ale, "tess_gpu_dual")) {
      numerics.ale.tess_gpu_dual =
          strict_bool(ale["tess_gpu_dual"], "Numerics.ale.tess_gpu_dual");
    }
    if (has_key(ale, "tess_gpu_restrict")) {
      numerics.ale.tess_gpu_restrict = strict_bool(
          ale["tess_gpu_restrict"], "Numerics.ale.tess_gpu_restrict");
    }
    if (has_key(ale, "dvclp_solver_rev")) {
      numerics.ale.dvclp_solver_rev = strict_int32(
          ale["dvclp_solver_rev"], "Numerics.ale.dvclp_solver_rev");
      if (numerics.ale.dvclp_solver_rev < 0 ||
          numerics.ale.dvclp_solver_rev > 1) {
        throw ConfigError(
            "Numerics.ale.dvclp_solver_rev must be 0 or 1");
      }
    }
    if (has_key(ale, "reale_lloyd_max")) {
      numerics.ale.reale_lloyd_max = strict_int32(
          ale["reale_lloyd_max"], "Numerics.ale.reale_lloyd_max");
      if (numerics.ale.reale_lloyd_max < 0 ||
          numerics.ale.reale_lloyd_max > 64) {
        throw ConfigError(
            "Numerics.ale.reale_lloyd_max must be in [0, 64]");
      }
    }
    if (has_key(ale, "reale_short_edge_collapse_rel")) {
      numerics.ale.reale_short_edge_collapse_rel = numeric_as_double(
          ale["reale_short_edge_collapse_rel"],
          "Numerics.ale.reale_short_edge_collapse_rel");
    }
    if (has_key(ale, "reale_subdomain_rezone")) {
      numerics.ale.reale_subdomain_rezone = strict_bool(
          ale["reale_subdomain_rezone"],
          "Numerics.ale.reale_subdomain_rezone");
    }
    if (has_key(ale, "reale_subdomain_frac_max")) {
      numerics.ale.reale_subdomain_frac_max = numeric_as_double(
          ale["reale_subdomain_frac_max"],
          "Numerics.ale.reale_subdomain_frac_max");
    }
    if (has_key(ale, "reale_overlay_additivity_tol")) {
      numerics.ale.reale_overlay_additivity_tol = numeric_as_double(
          ale["reale_overlay_additivity_tol"],
          "Numerics.ale.reale_overlay_additivity_tol");
    }
    if (has_key(ale, "reale_corner_mass_reset")) {
      numerics.ale.reale_corner_mass_reset = strict_bool(
          ale["reale_corner_mass_reset"],
          "Numerics.ale.reale_corner_mass_reset");
    }
    if (has_key(ale, "reale_velocity_max_principle")) {
      numerics.ale.reale_velocity_max_principle = strict_bool(
          ale["reale_velocity_max_principle"],
          "Numerics.ale.reale_velocity_max_principle");
    }
    if (has_key(ale, "reale_dt_trigger_factor")) {
      numerics.ale.reale_dt_trigger_factor = numeric_as_double(
          ale["reale_dt_trigger_factor"],
          "Numerics.ale.reale_dt_trigger_factor");
      if (!(numerics.ale.reale_dt_trigger_factor > 0.0) ||
          !(numerics.ale.reale_dt_trigger_factor <= 1.0)) {
        throw ConfigError(
            "Numerics.ale.reale_dt_trigger_factor must be in (0, 1]");
      }
    }
    if (has_key(ale, "reale_dt_trigger_cooldown")) {
      numerics.ale.reale_dt_trigger_cooldown = strict_int32(
          ale["reale_dt_trigger_cooldown"],
          "Numerics.ale.reale_dt_trigger_cooldown");
      if (numerics.ale.reale_dt_trigger_cooldown < 0 ||
          numerics.ale.reale_dt_trigger_cooldown > 100000) {
        throw ConfigError(
            "Numerics.ale.reale_dt_trigger_cooldown must be in "
            "[0, 100000]");
      }
    }
    if (has_key(ale, "ale_identity_mode")) {
      numerics.ale.ale_identity_mode = strict_bool(
          ale["ale_identity_mode"], "Numerics.ale.ale_identity_mode");
    }
    if (has_key(ale, "ale_mover_diag")) {
      numerics.ale.ale_mover_diag = strict_bool(
          ale["ale_mover_diag"], "Numerics.ale.ale_mover_diag");
    }
    if (has_key(ale, "ale_preserve_lagrangian_velocity_carry")) {
      numerics.ale.ale_preserve_lagrangian_velocity_carry = strict_bool(
          ale["ale_preserve_lagrangian_velocity_carry"],
          "Numerics.ale.ale_preserve_lagrangian_velocity_carry");
    }
    if (has_key(ale, "align_diagnostics")) {
      const py::handle align_obj = ale["align_diagnostics"];
      if (!py::isinstance<py::dict>(align_obj)) {
        throw_value_type_error("Numerics.ale.align_diagnostics", "dict",
                               align_obj);
      }
      const py::dict align = py::reinterpret_borrow<py::dict>(align_obj);
      enforce_known_keys(align, "Numerics.ale.align_diagnostics",
                         {"enabled", "every_n_steps", "c_q_threshold",
                          "w_rho", "w_p", "floor_rel"});
      auto& align_cfg = numerics.ale.align_diagnostics;
      if (has_key(align, "enabled")) {
        align_cfg.enabled = strict_bool(
            align["enabled"], "Numerics.ale.align_diagnostics.enabled");
      }
      if (has_key(align, "every_n_steps")) {
        align_cfg.every_n_steps = strict_int32(
            align["every_n_steps"],
            "Numerics.ale.align_diagnostics.every_n_steps");
      }
      if (has_key(align, "c_q_threshold")) {
        align_cfg.c_q_threshold = numeric_as_double(
            align["c_q_threshold"],
            "Numerics.ale.align_diagnostics.c_q_threshold");
      }
      if (has_key(align, "w_rho")) {
        align_cfg.w_rho = numeric_as_double(
            align["w_rho"], "Numerics.ale.align_diagnostics.w_rho");
      }
      if (has_key(align, "w_p")) {
        align_cfg.w_p = numeric_as_double(
            align["w_p"], "Numerics.ale.align_diagnostics.w_p");
      }
      if (has_key(align, "floor_rel")) {
        align_cfg.floor_rel = numeric_as_double(
            align["floor_rel"],
            "Numerics.ale.align_diagnostics.floor_rel");
      }
    }
    const bool swept_volume_sign_fixed_explicit =
        has_key(ale, "swept_volume_sign_fixed");
    if (swept_volume_sign_fixed_explicit) {
      numerics.ale.swept_volume_sign_fixed =
          strict_bool(ale["swept_volume_sign_fixed"],
                      "Numerics.ale.swept_volume_sign_fixed");
    }
    if (has_key(ale, "donor_sign_fixed")) {
      tenryu::core::log_warning(
          "Numerics.ale.donor_sign_fixed is deprecated; use "
          "Numerics.ale.swept_volume_sign_fixed");
      const bool alias_value =
          strict_bool(ale["donor_sign_fixed"], "Numerics.ale.donor_sign_fixed");
      if (swept_volume_sign_fixed_explicit &&
          alias_value != numerics.ale.swept_volume_sign_fixed) {
        throw ConfigError(
            "Numerics.ale.donor_sign_fixed conflicts with "
            "Numerics.ale.swept_volume_sign_fixed");
      }
      if (!swept_volume_sign_fixed_explicit) {
        numerics.ale.swept_volume_sign_fixed = alias_value;
      }
    }
    if (!numerics.ale.swept_volume_sign_fixed) {
      throw ConfigError(
          "legacy swept-volume sign convention removed 2026-08-05 (epoch "
          "2); see NUMERICS");
    }
    if (has_key(ale, "every_n_steps")) {
      numerics.ale.every_n_steps =
          strict_int32(ale["every_n_steps"], "Numerics.ale.every_n_steps");
    }
    if (has_key(ale, "force_rezone_every_n_steps")) {
      numerics.ale.force_rezone_every_n_steps = strict_int32(
          ale["force_rezone_every_n_steps"],
          "Numerics.ale.force_rezone_every_n_steps");
    }
    if (has_key(ale, "warmup_steps")) {
      numerics.ale.warmup_steps =
          strict_int32(ale["warmup_steps"], "Numerics.ale.warmup_steps");
    }
    if (has_key(ale, "relaxation")) {
      numerics.ale.relaxation =
          numeric_as_double(ale["relaxation"], "Numerics.ale.relaxation");
    }
    if (has_key(ale, "spacing_ratio_threshold")) {
      numerics.ale.spacing_ratio_threshold = numeric_as_double(
          ale["spacing_ratio_threshold"], "Numerics.ale.spacing_ratio_threshold");
    }
    if (has_key(ale, "quality_threshold")) {
      numerics.ale.quality_threshold =
          numeric_as_double(ale["quality_threshold"], "Numerics.ale.quality_threshold");
    }
    if (has_key(ale, "max_iterations")) {
      numerics.ale.max_iterations =
          strict_int32(ale["max_iterations"], "Numerics.ale.max_iterations");
    }
    if (has_key(ale, "convergence_tol")) {
      numerics.ale.convergence_tol =
          numeric_as_double(ale["convergence_tol"], "Numerics.ale.convergence_tol");
    }
    if (has_key(ale, "max_displacement_fraction")) {
      numerics.ale.max_displacement_fraction = numeric_as_double(
          ale["max_displacement_fraction"], "Numerics.ale.max_displacement_fraction");
    }
    if (remap_limiter_explicit) {
      numerics.ale.remap_limiter =
          strict_string(ale["remap_limiter"], "Numerics.ale.remap_limiter");
      if (!is_ale_remap_limiter(numerics.ale.remap_limiter)) {
        throw ValueError(
            "Numerics.ale.remap_limiter must be one of {\"van_leer\", \"minmod\"}");
      }
    }
    if (remap_ms_midpoint_explicit) {
      numerics.ale.remap_ms_midpoint =
          strict_bool(ale["remap_ms_midpoint"], "Numerics.ale.remap_ms_midpoint");
    }
    if (remap_ms_post_check_explicit) {
      numerics.ale.remap_ms_post_check =
          strict_bool(ale["remap_ms_post_check"], "Numerics.ale.remap_ms_post_check");
    }
    if (has_key(ale, "remap_ms_post_max_iter")) {
      numerics.ale.remap_ms_post_max_iter = strict_int32(
          ale["remap_ms_post_max_iter"], "Numerics.ale.remap_ms_post_max_iter");
    }
    if (has_key(ale, "remap_ms_rescale_floor")) {
      numerics.ale.remap_ms_rescale_floor = numeric_as_double(
          ale["remap_ms_rescale_floor"], "Numerics.ale.remap_ms_rescale_floor");
    }
    const bool remap_ms_enabled =
        numerics.ale.remap_limiter == "minmod" || numerics.ale.remap_ms_midpoint ||
        numerics.ale.remap_ms_post_check;
    if (remap_ms_enabled) {
      if (!remap_limiter_explicit) {
        numerics.ale.remap_limiter = "minmod";
      }
      if (!remap_ms_midpoint_explicit) {
        numerics.ale.remap_ms_midpoint = true;
      }
    }
    if (has_key(ale, "ke_fixup")) {
      numerics.ale.ke_fixup = strict_bool(ale["ke_fixup"], "Numerics.ale.ke_fixup");
    }
    if (has_key(ale, "ke_conservation_closure")) {
      numerics.ale.ke_conservation_closure = strict_bool(
          ale["ke_conservation_closure"], "Numerics.ale.ke_conservation_closure");
    }
    if (has_key(ale, "ke_conservation_closure_audit")) {
      numerics.ale.ke_conservation_closure_audit = strict_bool(
          ale["ke_conservation_closure_audit"],
          "Numerics.ale.ke_conservation_closure_audit");
    }
    if (has_key(ale, "ke_closure_redistribute_floor")) {
      numerics.ale.ke_closure_redistribute_floor = strict_bool(
          ale["ke_closure_redistribute_floor"],
          "Numerics.ale.ke_closure_redistribute_floor");
    }
    if (has_key(ale, "debug_per_remap_log")) {
      numerics.ale.debug_per_remap_log = strict_bool(
          ale["debug_per_remap_log"], "Numerics.ale.debug_per_remap_log");
    }
    if (has_key(ale, "shock_sensor_guard_cells")) {
      numerics.ale.shock_sensor_guard_cells = strict_int32(
          ale["shock_sensor_guard_cells"], "Numerics.ale.shock_sensor_guard_cells");
    }
    if (has_key(ale, "density_jump_threshold")) {
      numerics.ale.density_jump_threshold = numeric_as_double(
          ale["density_jump_threshold"], "Numerics.ale.density_jump_threshold");
    }
    if (has_key(ale, "Te_jump_threshold")) {
      numerics.ale.Te_jump_threshold =
          numeric_as_double(ale["Te_jump_threshold"], "Numerics.ale.Te_jump_threshold");
    }
    if (has_key(ale, "preventive_axis_guard_fraction")) {
      numerics.ale.preventive_axis_guard_fraction = numeric_as_double(
          ale["preventive_axis_guard_fraction"],
          "Numerics.ale.preventive_axis_guard_fraction");
    }
    if (has_key(ale, "axis_z_motion")) {
      numerics.ale.axis_z_motion =
          strict_string(ale["axis_z_motion"], "Numerics.ale.axis_z_motion");
      if (numerics.ale.axis_z_motion != "fixed" &&
          numerics.ale.axis_z_motion != "winslow" &&
          numerics.ale.axis_z_motion != "lagrangian" &&
          numerics.ale.axis_z_motion != "lagrangian_tangential") {
        throw ValueError(
            "Numerics.ale.axis_z_motion must be one of {\"fixed\", \"winslow\", "
            "\"lagrangian\", \"lagrangian_tangential\"}");
      }
      if (numerics.ale.axis_z_motion == "lagrangian") {
        throw ValueError(
            "Numerics.ale.axis_z_motion=\"lagrangian\" is not yet implemented "
            "(Phase 8b future work) — use \"lagrangian_tangential\" instead");
      }
    }
    if (has_key(ale, "winslow_axis_kappa")) {
      numerics.ale.winslow_axis_kappa = numeric_as_double(
          ale["winslow_axis_kappa"], "Numerics.ale.winslow_axis_kappa");
    }
    if (has_key(ale, "button_morph")) {
      const py::handle button_morph_obj = ale["button_morph"];
      if (!py::isinstance<py::dict>(button_morph_obj)) {
        throw_value_type_error("Numerics.ale.button_morph", "dict",
                               button_morph_obj);
      }
      const py::dict button_morph =
          py::reinterpret_borrow<py::dict>(button_morph_obj);
      enforce_known_keys(button_morph, "Numerics.ale.button_morph",
                         {"enabled", "t_start_s", "t_end_s",
                          "max_step_fraction", "every_n_steps"});
      auto& button_morph_cfg = numerics.ale.button_morph;
      if (has_key(button_morph, "enabled")) {
        button_morph_cfg.enabled = strict_bool(
            button_morph["enabled"], "Numerics.ale.button_morph.enabled");
      }
      if (has_key(button_morph, "t_start_s")) {
        button_morph_cfg.t_start_s = numeric_as_double(
            button_morph["t_start_s"],
            "Numerics.ale.button_morph.t_start_s");
      }
      if (has_key(button_morph, "t_end_s")) {
        button_morph_cfg.t_end_s = numeric_as_double(
            button_morph["t_end_s"],
            "Numerics.ale.button_morph.t_end_s");
      }
      if (has_key(button_morph, "max_step_fraction")) {
        button_morph_cfg.max_step_fraction = numeric_as_double(
            button_morph["max_step_fraction"],
            "Numerics.ale.button_morph.max_step_fraction");
      }
      if (has_key(button_morph, "every_n_steps")) {
        button_morph_cfg.every_n_steps = strict_int32(
            button_morph["every_n_steps"],
            "Numerics.ale.button_morph.every_n_steps");
      }
      ensure_int_ge(button_morph_cfg.every_n_steps, 1,
                    "Numerics.ale.button_morph.every_n_steps");
      if (!(std::isfinite(button_morph_cfg.max_step_fraction) &&
            button_morph_cfg.max_step_fraction > 0.0 &&
            button_morph_cfg.max_step_fraction <= 0.5)) {
        throw ValueError(
            "Numerics.ale.button_morph.max_step_fraction must be finite and in (0, 0.5]");
      }
      if (!(std::isfinite(button_morph_cfg.t_start_s) &&
            button_morph_cfg.t_start_s >= 0.0)) {
        throw ValueError(
            "Numerics.ale.button_morph.t_start_s must be finite and >= 0");
      }
      if (!(std::isfinite(button_morph_cfg.t_end_s) &&
            button_morph_cfg.t_end_s >= 0.0)) {
        throw ValueError(
            "Numerics.ale.button_morph.t_end_s must be finite and >= 0");
      }
      if (button_morph_cfg.enabled &&
          !(button_morph_cfg.t_end_s > button_morph_cfg.t_start_s)) {
        throw ConfigError(
            "Numerics.ale.button_morph.t_end_s must exceed t_start_s when enabled");
      }
    }
    if (has_key(ale, "runtime_controller")) {
      const py::handle runtime_controller_obj = ale["runtime_controller"];
      if (!py::isinstance<py::dict>(runtime_controller_obj)) {
        throw_value_type_error("Numerics.ale.runtime_controller", "dict",
                               runtime_controller_obj);
      }
      const py::dict runtime_controller =
          py::reinterpret_borrow<py::dict>(runtime_controller_obj);
      enforce_known_keys(runtime_controller,
                         "Numerics.ale.runtime_controller",
                         {"monitor_enabled", "monitor_every", "shell_rows",
                          "controller_shell_rows", "cap_columns",
                          "q_soft", "q_hard", "q_recover", "h_soft",
                          "h_hard", "h_recover", "soft_persistence",
                          "recover_checks", "winslow_sweeps",
                          "winslow_omega", "beta_monitor_soft",
                          "beta_monitor_hard", "beta_mass", "beta_front",
                          "beta_theta", "g_max", "front_width_cells",
                          "cap_fraction", "cap_normal_fraction",
                          "controller_enabled", "commit_rollback_enabled",
                          "activation_front_mode",
                          "activation_front_margin_hs", "activation_time_s",
                          "cadence_soft",
                          "cadence_hard", "cadence_recovery",
                          "pre_step_enabled", "failures_hard_force",
                          "failures_big_repair", "escalation_max_failures"});
      auto& runtime_cfg = numerics.ale.runtime_controller;
      if (has_key(runtime_controller, "monitor_enabled")) {
        runtime_cfg.monitor_enabled = strict_bool(
            runtime_controller["monitor_enabled"],
            "Numerics.ale.runtime_controller.monitor_enabled");
      }
      if (has_key(runtime_controller, "monitor_every")) {
        runtime_cfg.monitor_every = strict_int32(
            runtime_controller["monitor_every"],
            "Numerics.ale.runtime_controller.monitor_every");
      }
      if (has_key(runtime_controller, "shell_rows")) {
        runtime_cfg.shell_rows = strict_int32(
            runtime_controller["shell_rows"],
            "Numerics.ale.runtime_controller.shell_rows");
      }
      if (has_key(runtime_controller, "controller_shell_rows")) {
        runtime_cfg.controller_shell_rows = strict_int32(
            runtime_controller["controller_shell_rows"],
            "Numerics.ale.runtime_controller.controller_shell_rows");
      }
      if (has_key(runtime_controller, "cap_columns")) {
        runtime_cfg.cap_columns = strict_int32(
            runtime_controller["cap_columns"],
            "Numerics.ale.runtime_controller.cap_columns");
      }
      if (has_key(runtime_controller, "q_soft")) {
        runtime_cfg.q_soft = numeric_as_double(
            runtime_controller["q_soft"],
            "Numerics.ale.runtime_controller.q_soft");
      }
      if (has_key(runtime_controller, "q_hard")) {
        runtime_cfg.q_hard = numeric_as_double(
            runtime_controller["q_hard"],
            "Numerics.ale.runtime_controller.q_hard");
      }
      if (has_key(runtime_controller, "q_recover")) {
        runtime_cfg.q_recover = numeric_as_double(
            runtime_controller["q_recover"],
            "Numerics.ale.runtime_controller.q_recover");
      }
      if (has_key(runtime_controller, "h_soft")) {
        runtime_cfg.h_soft = numeric_as_double(
            runtime_controller["h_soft"],
            "Numerics.ale.runtime_controller.h_soft");
      }
      if (has_key(runtime_controller, "h_hard")) {
        runtime_cfg.h_hard = numeric_as_double(
            runtime_controller["h_hard"],
            "Numerics.ale.runtime_controller.h_hard");
      }
      if (has_key(runtime_controller, "h_recover")) {
        runtime_cfg.h_recover = numeric_as_double(
            runtime_controller["h_recover"],
            "Numerics.ale.runtime_controller.h_recover");
      }
      if (has_key(runtime_controller, "soft_persistence")) {
        runtime_cfg.soft_persistence = strict_int32(
            runtime_controller["soft_persistence"],
            "Numerics.ale.runtime_controller.soft_persistence");
      }
      if (has_key(runtime_controller, "recover_checks")) {
        runtime_cfg.recover_checks = strict_int32(
            runtime_controller["recover_checks"],
            "Numerics.ale.runtime_controller.recover_checks");
      }
      if (has_key(runtime_controller, "winslow_sweeps")) {
        runtime_cfg.winslow_sweeps = strict_int32(
            runtime_controller["winslow_sweeps"],
            "Numerics.ale.runtime_controller.winslow_sweeps");
      }
      if (has_key(runtime_controller, "winslow_omega")) {
        runtime_cfg.winslow_omega = numeric_as_double(
            runtime_controller["winslow_omega"],
            "Numerics.ale.runtime_controller.winslow_omega");
      }
      if (has_key(runtime_controller, "beta_monitor_soft")) {
        runtime_cfg.beta_monitor_soft = numeric_as_double(
            runtime_controller["beta_monitor_soft"],
            "Numerics.ale.runtime_controller.beta_monitor_soft");
      }
      if (has_key(runtime_controller, "beta_monitor_hard")) {
        runtime_cfg.beta_monitor_hard = numeric_as_double(
            runtime_controller["beta_monitor_hard"],
            "Numerics.ale.runtime_controller.beta_monitor_hard");
      }
      if (has_key(runtime_controller, "beta_mass")) {
        runtime_cfg.beta_mass = numeric_as_double(
            runtime_controller["beta_mass"],
            "Numerics.ale.runtime_controller.beta_mass");
      }
      if (has_key(runtime_controller, "beta_front")) {
        runtime_cfg.beta_front = numeric_as_double(
            runtime_controller["beta_front"],
            "Numerics.ale.runtime_controller.beta_front");
      }
      if (has_key(runtime_controller, "beta_theta")) {
        runtime_cfg.beta_theta = numeric_as_double(
            runtime_controller["beta_theta"],
            "Numerics.ale.runtime_controller.beta_theta");
      }
      if (has_key(runtime_controller, "g_max")) {
        runtime_cfg.g_max = numeric_as_double(
            runtime_controller["g_max"],
            "Numerics.ale.runtime_controller.g_max");
      }
      if (has_key(runtime_controller, "front_width_cells")) {
        runtime_cfg.front_width_cells = numeric_as_double(
            runtime_controller["front_width_cells"],
            "Numerics.ale.runtime_controller.front_width_cells");
      }
      if (has_key(runtime_controller, "cap_fraction")) {
        runtime_cfg.cap_fraction = numeric_as_double(
            runtime_controller["cap_fraction"],
            "Numerics.ale.runtime_controller.cap_fraction");
      }
      if (has_key(runtime_controller, "cap_normal_fraction")) {
        runtime_cfg.cap_normal_fraction = numeric_as_double(
            runtime_controller["cap_normal_fraction"],
            "Numerics.ale.runtime_controller.cap_normal_fraction");
      }
      if (has_key(runtime_controller, "controller_enabled")) {
        runtime_cfg.controller_enabled = strict_bool(
            runtime_controller["controller_enabled"],
            "Numerics.ale.runtime_controller.controller_enabled");
      }
      if (has_key(runtime_controller, "commit_rollback_enabled")) {
        runtime_cfg.commit_rollback_enabled = strict_bool(
            runtime_controller["commit_rollback_enabled"],
            "Numerics.ale.runtime_controller.commit_rollback_enabled");
      }
      if (has_key(runtime_controller, "activation_front_mode")) {
        runtime_cfg.activation_front_mode = strict_string(
            runtime_controller["activation_front_mode"],
            "Numerics.ale.runtime_controller.activation_front_mode");
      }
      if (has_key(runtime_controller, "activation_front_margin_hs")) {
        runtime_cfg.activation_front_margin_hs = numeric_as_double(
            runtime_controller["activation_front_margin_hs"],
            "Numerics.ale.runtime_controller.activation_front_margin_hs");
      }
      if (has_key(runtime_controller, "activation_time_s")) {
        runtime_cfg.activation_time_s = numeric_as_double(
            runtime_controller["activation_time_s"],
            "Numerics.ale.runtime_controller.activation_time_s");
      }
      if (has_key(runtime_controller, "cadence_soft")) {
        runtime_cfg.cadence_soft = strict_int32(
            runtime_controller["cadence_soft"],
            "Numerics.ale.runtime_controller.cadence_soft");
      }
      if (has_key(runtime_controller, "cadence_hard")) {
        runtime_cfg.cadence_hard = strict_int32(
            runtime_controller["cadence_hard"],
            "Numerics.ale.runtime_controller.cadence_hard");
      }
      if (has_key(runtime_controller, "cadence_recovery")) {
        runtime_cfg.cadence_recovery = strict_int32(
            runtime_controller["cadence_recovery"],
            "Numerics.ale.runtime_controller.cadence_recovery");
      }
      if (has_key(runtime_controller, "pre_step_enabled")) {
        runtime_cfg.pre_step_enabled = strict_bool(
            runtime_controller["pre_step_enabled"],
            "Numerics.ale.runtime_controller.pre_step_enabled");
      }
      if (has_key(runtime_controller, "failures_hard_force")) {
        runtime_cfg.failures_hard_force = strict_int32(
            runtime_controller["failures_hard_force"],
            "Numerics.ale.runtime_controller.failures_hard_force");
      }
      if (has_key(runtime_controller, "failures_big_repair")) {
        runtime_cfg.failures_big_repair = strict_int32(
            runtime_controller["failures_big_repair"],
            "Numerics.ale.runtime_controller.failures_big_repair");
      }
      if (has_key(runtime_controller, "escalation_max_failures")) {
        runtime_cfg.escalation_max_failures = strict_int32(
            runtime_controller["escalation_max_failures"],
            "Numerics.ale.runtime_controller.escalation_max_failures");
      }

      ensure_int_ge(runtime_cfg.monitor_every, 1,
                    "Numerics.ale.runtime_controller.monitor_every");
      ensure_int_ge(runtime_cfg.shell_rows, 0,
                    "Numerics.ale.runtime_controller.shell_rows");
      ensure_int_ge(runtime_cfg.controller_shell_rows, 0,
                    "Numerics.ale.runtime_controller.controller_shell_rows");
      ensure_int_ge(runtime_cfg.cap_columns, 0,
                    "Numerics.ale.runtime_controller.cap_columns");
      ensure_int_ge(runtime_cfg.soft_persistence, 1,
                    "Numerics.ale.runtime_controller.soft_persistence");
      ensure_int_ge(runtime_cfg.recover_checks, 1,
                    "Numerics.ale.runtime_controller.recover_checks");
      ensure_int_ge(runtime_cfg.cadence_soft, 1,
                    "Numerics.ale.runtime_controller.cadence_soft");
      ensure_int_ge(runtime_cfg.cadence_hard, 1,
                    "Numerics.ale.runtime_controller.cadence_hard");
      ensure_int_ge(runtime_cfg.cadence_recovery, 1,
                    "Numerics.ale.runtime_controller.cadence_recovery");
      if (runtime_cfg.failures_hard_force < 1 ||
          runtime_cfg.failures_big_repair < runtime_cfg.failures_hard_force ||
          runtime_cfg.escalation_max_failures <
              runtime_cfg.failures_big_repair) {
        throw ValueError(
            "Numerics.ale.runtime_controller failure thresholds must satisfy "
            "1 <= failures_hard_force <= failures_big_repair <= "
            "escalation_max_failures");
      }
      if (runtime_cfg.cadence_hard > runtime_cfg.cadence_soft) {
        throw ValueError(
            "Numerics.ale.runtime_controller.cadence_hard must be <= cadence_soft");
      }
      if (runtime_cfg.cadence_recovery < runtime_cfg.cadence_soft) {
        throw ValueError(
            "Numerics.ale.runtime_controller.cadence_recovery must be >= cadence_soft");
      }
      if (runtime_cfg.controller_enabled && !runtime_cfg.monitor_enabled) {
        throw ValueError(
            "Numerics.ale.runtime_controller.controller_enabled=true requires monitor_enabled=true");
      }
      if (runtime_cfg.activation_front_mode != "min" &&
          runtime_cfg.activation_front_mode != "mean") {
        throw ValueError(
            "Numerics.ale.runtime_controller.activation_front_mode must be one of {\"min\", \"mean\"}");
      }
      if (!(std::isfinite(runtime_cfg.activation_front_margin_hs) &&
            runtime_cfg.activation_front_margin_hs >= 0.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.activation_front_margin_hs must be finite and >= 0");
      }
      if (runtime_cfg.winslow_sweeps < 1 || runtime_cfg.winslow_sweeps > 16) {
        throw ValueError(
            "Numerics.ale.runtime_controller.winslow_sweeps must be in [1, 16]");
      }
      if (!(std::isfinite(runtime_cfg.q_soft) && runtime_cfg.q_soft > 0.0 &&
            runtime_cfg.q_soft <= 1.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.q_soft must be finite and in (0, 1]");
      }
      if (!(std::isfinite(runtime_cfg.q_hard) && runtime_cfg.q_hard > 0.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.q_hard must be finite and > 0");
      }
      if (!(std::isfinite(runtime_cfg.q_recover) &&
            runtime_cfg.q_recover > 0.0 && runtime_cfg.q_recover <= 1.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.q_recover must be finite and in (0, 1]");
      }
      if (!(runtime_cfg.q_hard < runtime_cfg.q_soft)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.q_hard must be less than q_soft");
      }
      if (!(runtime_cfg.q_soft < runtime_cfg.q_recover)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.q_recover must exceed q_soft");
      }
      if (!(std::isfinite(runtime_cfg.h_soft) && runtime_cfg.h_soft > 0.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.h_soft must be finite and > 0");
      }
      if (!(std::isfinite(runtime_cfg.h_hard) && runtime_cfg.h_hard > 0.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.h_hard must be finite and > 0");
      }
      if (!(std::isfinite(runtime_cfg.h_recover) &&
            runtime_cfg.h_recover > 0.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.h_recover must be finite and > 0");
      }
      if (!(runtime_cfg.h_hard < runtime_cfg.h_soft)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.h_hard must be less than h_soft");
      }
      if (!(runtime_cfg.h_soft < runtime_cfg.h_recover)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.h_recover must exceed h_soft");
      }
      if (!(std::isfinite(runtime_cfg.winslow_omega) &&
            runtime_cfg.winslow_omega > 0.0 &&
            runtime_cfg.winslow_omega <= 1.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.winslow_omega must be finite and in (0, 1]");
      }
      if (!(std::isfinite(runtime_cfg.beta_monitor_soft) &&
            runtime_cfg.beta_monitor_soft >= 0.0 &&
            runtime_cfg.beta_monitor_soft < 1.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.beta_monitor_soft must be finite and in [0, 1)");
      }
      if (!(std::isfinite(runtime_cfg.beta_monitor_hard) &&
            runtime_cfg.beta_monitor_hard >= 0.0 &&
            runtime_cfg.beta_monitor_hard < 1.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.beta_monitor_hard must be finite and in [0, 1)");
      }
      if (!(std::isfinite(runtime_cfg.beta_mass) &&
            runtime_cfg.beta_mass >= 0.0 && runtime_cfg.beta_mass <= 0.5)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.beta_mass must be finite and in [0, 0.5]");
      }
      if (!(std::isfinite(runtime_cfg.beta_front) &&
            runtime_cfg.beta_front >= 0.0 && runtime_cfg.beta_front <= 0.5)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.beta_front must be finite and in [0, 0.5]");
      }
      if (runtime_cfg.beta_mass + runtime_cfg.beta_front > 0.9) {
        throw ValueError(
            "Numerics.ale.runtime_controller.beta_mass + beta_front must be <= 0.9");
      }
      if (!(std::isfinite(runtime_cfg.beta_theta) &&
            runtime_cfg.beta_theta >= 0.0 && runtime_cfg.beta_theta <= 0.5)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.beta_theta must be finite and in [0, 0.5]");
      }
      if (!(std::isfinite(runtime_cfg.g_max) && runtime_cfg.g_max > 1.0 &&
            runtime_cfg.g_max < 1.6)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.g_max must be finite and in (1, 1.6)");
      }
      if (!(std::isfinite(runtime_cfg.front_width_cells) &&
            runtime_cfg.front_width_cells >= 1.0 &&
            runtime_cfg.front_width_cells <= 8.0)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.front_width_cells must be finite and in [1, 8]");
      }
      if (!(std::isfinite(runtime_cfg.cap_fraction) &&
            runtime_cfg.cap_fraction > 0.0 &&
            runtime_cfg.cap_fraction <= 0.2)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.cap_fraction must be finite and in (0, 0.2]");
      }
      if (!(std::isfinite(runtime_cfg.cap_normal_fraction) &&
            runtime_cfg.cap_normal_fraction > 0.0 &&
            runtime_cfg.cap_normal_fraction <= runtime_cfg.cap_fraction)) {
        throw ValueError(
            "Numerics.ale.runtime_controller.cap_normal_fraction must be finite and in (0, cap_fraction]");
      }
    }
    if (has_key(ale, "reference_barrier_enabled")) {
      numerics.ale.reference_barrier_enabled = strict_bool(
          ale["reference_barrier_enabled"],
          "Numerics.ale.reference_barrier_enabled");
    }
    if (has_key(ale, "reference_target")) {
      numerics.ale.reference_target =
          strict_string(ale["reference_target"], "Numerics.ale.reference_target");
      if (numerics.ale.reference_target != "none" &&
          numerics.ale.reference_target != "eulerian_initial" &&
          numerics.ale.reference_target != "spherical_equal_angle") {
        throw ValueError(
            "Numerics.ale.reference_target must be one of {\"none\", "
            "\"eulerian_initial\", \"spherical_equal_angle\"}");
      }
    }
    if (has_key(ale, "reference_blend_default")) {
      numerics.ale.reference_blend_default = numeric_as_double(
          ale["reference_blend_default"], "Numerics.ale.reference_blend_default");
    }
    if (has_key(ale, "reference_volume_floor_rel")) {
      numerics.ale.reference_volume_floor_rel = numeric_as_double(
          ale["reference_volume_floor_rel"], "Numerics.ale.reference_volume_floor_rel");
    }
    if (has_key(ale, "reference_corner_j_floor_rel")) {
      numerics.ale.reference_corner_j_floor_rel = numeric_as_double(
          ale["reference_corner_j_floor_rel"],
          "Numerics.ale.reference_corner_j_floor_rel");
    }
    if (has_key(ale, "reference_gauss_j_floor_rel")) {
      numerics.ale.reference_gauss_j_floor_rel = numeric_as_double(
          ale["reference_gauss_j_floor_rel"],
          "Numerics.ale.reference_gauss_j_floor_rel");
    }
    if (has_key(ale, "reference_linesearch_max_iters")) {
      numerics.ale.reference_linesearch_max_iters = strict_int32(
          ale["reference_linesearch_max_iters"],
          "Numerics.ale.reference_linesearch_max_iters");
    }
    if (has_key(ale, "reference_force_engage_every_step")) {
      numerics.ale.reference_force_engage_every_step = strict_bool(
          ale["reference_force_engage_every_step"],
          "Numerics.ale.reference_force_engage_every_step");
    }
    if (has_key(ale, "reference_trigger_axis_margin_enabled")) {
      numerics.ale.reference_trigger_axis_margin_enabled = strict_bool(
          ale["reference_trigger_axis_margin_enabled"],
          "Numerics.ale.reference_trigger_axis_margin_enabled");
    }
    if (has_key(ale, "reference_trigger_axis_margin_threshold")) {
      numerics.ale.reference_trigger_axis_margin_threshold = numeric_as_double(
          ale["reference_trigger_axis_margin_threshold"],
          "Numerics.ale.reference_trigger_axis_margin_threshold");
    }
    if (has_key(ale, "reference_trigger_corner_j_ratio_enabled")) {
      numerics.ale.reference_trigger_corner_j_ratio_enabled = strict_bool(
          ale["reference_trigger_corner_j_ratio_enabled"],
          "Numerics.ale.reference_trigger_corner_j_ratio_enabled");
    }
    if (has_key(ale, "reference_trigger_corner_j_ratio_threshold")) {
      numerics.ale.reference_trigger_corner_j_ratio_threshold = numeric_as_double(
          ale["reference_trigger_corner_j_ratio_threshold"],
          "Numerics.ale.reference_trigger_corner_j_ratio_threshold");
    }
    if (has_key(ale, "dgcl_commit_gate")) {
      numerics.ale.dgcl_commit_gate = strict_bool(
          ale["dgcl_commit_gate"], "Numerics.ale.dgcl_commit_gate");
    }
    if (has_key(ale, "transaction_failure_inject_point")) {
      numerics.ale.transaction_failure_inject_point = strict_int32(
          ale["transaction_failure_inject_point"],
          "Numerics.ale.transaction_failure_inject_point");
    }
    if (has_key(ale, "dgcl_commit_rtol")) {
      numerics.ale.dgcl_commit_rtol = numeric_as_double(
          ale["dgcl_commit_rtol"], "Numerics.ale.dgcl_commit_rtol");
    }
    if (has_key(ale, "driver_retry_reference_barrier_enabled")) {
      numerics.ale.driver_retry_reference_barrier_enabled = strict_bool(
          ale["driver_retry_reference_barrier_enabled"],
          "Numerics.ale.driver_retry_reference_barrier_enabled");
    }
    if (has_key(ale, "driver_retry_reference_barrier_K_axis")) {
      numerics.ale.driver_retry_reference_barrier_K_axis = strict_int32(
          ale["driver_retry_reference_barrier_K_axis"],
          "Numerics.ale.driver_retry_reference_barrier_K_axis");
    }
    if (has_key(ale, "driver_retry_reference_barrier_eta_axis")) {
      numerics.ale.driver_retry_reference_barrier_eta_axis = numeric_as_double(
          ale["driver_retry_reference_barrier_eta_axis"],
          "Numerics.ale.driver_retry_reference_barrier_eta_axis");
    }
    if (has_key(ale, "driver_retry_reference_barrier_max_attempts")) {
      numerics.ale.driver_retry_reference_barrier_max_attempts = strict_int32(
          ale["driver_retry_reference_barrier_max_attempts"],
          "Numerics.ale.driver_retry_reference_barrier_max_attempts");
    }
    if (has_key(ale, "driver_retry_reference_barrier_same_sig_max")) {
      numerics.ale.driver_retry_reference_barrier_same_sig_max = strict_int32(
          ale["driver_retry_reference_barrier_same_sig_max"],
          "Numerics.ale.driver_retry_reference_barrier_same_sig_max");
    }
    if (has_key(ale, "driver_retry_reference_barrier_cell_window")) {
      numerics.ale.driver_retry_reference_barrier_cell_window = strict_int32(
          ale["driver_retry_reference_barrier_cell_window"],
          "Numerics.ale.driver_retry_reference_barrier_cell_window");
    }
    if (has_key(ale, "driver_retry_reference_barrier_dt_collapse_rel")) {
      numerics.ale.driver_retry_reference_barrier_dt_collapse_rel =
          numeric_as_double(ale["driver_retry_reference_barrier_dt_collapse_rel"],
                            "Numerics.ale.driver_retry_reference_barrier_dt_collapse_rel");
    }
    if (has_key(ale, "driver_retry_reference_barrier_lambda_collapse_threshold")) {
      numerics.ale.driver_retry_reference_barrier_lambda_collapse_threshold =
          numeric_as_double(
              ale["driver_retry_reference_barrier_lambda_collapse_threshold"],
              "Numerics.ale.driver_retry_reference_barrier_lambda_collapse_threshold");
    }
    if (has_key(ale, "driver_retry_reference_barrier_lambda_collapse_count")) {
      numerics.ale.driver_retry_reference_barrier_lambda_collapse_count =
          strict_int32(
              ale["driver_retry_reference_barrier_lambda_collapse_count"],
              "Numerics.ale.driver_retry_reference_barrier_lambda_collapse_count");
    }
    if (has_key(ale, "driver_retry_reference_barrier_quality_progress_factor")) {
      numerics.ale.driver_retry_reference_barrier_quality_progress_factor =
          numeric_as_double(
              ale["driver_retry_reference_barrier_quality_progress_factor"],
              "Numerics.ale.driver_retry_reference_barrier_quality_progress_factor");
    }
    if (has_key(ale, "driver_retry_reference_barrier_quality_progress_count")) {
      numerics.ale.driver_retry_reference_barrier_quality_progress_count =
          strict_int32(
              ale["driver_retry_reference_barrier_quality_progress_count"],
              "Numerics.ale.driver_retry_reference_barrier_quality_progress_count");
    }
    if (has_key(ale, "driver_retry_reference_barrier_rezone_freq_warn_fraction")) {
      numerics.ale.driver_retry_reference_barrier_rezone_freq_warn_fraction =
          numeric_as_double(
              ale["driver_retry_reference_barrier_rezone_freq_warn_fraction"],
              "Numerics.ale.driver_retry_reference_barrier_rezone_freq_warn_fraction");
    }
    if (has_key(ale, "driver_retry_reference_barrier_rezone_freq_window")) {
      numerics.ale.driver_retry_reference_barrier_rezone_freq_window =
          strict_int32(ale["driver_retry_reference_barrier_rezone_freq_window"],
                       "Numerics.ale.driver_retry_reference_barrier_rezone_freq_window");
    }
    if (has_key(ale, "driver_retry_reference_barrier_chi")) {
      numerics.ale.driver_retry_reference_barrier_chi = numeric_as_double(
          ale["driver_retry_reference_barrier_chi"],
          "Numerics.ale.driver_retry_reference_barrier_chi");
    }
    if (has_key(ale, "driver_retry_reference_barrier_q_retry")) {
      numerics.ale.driver_retry_reference_barrier_q_retry = numeric_as_double(
          ale["driver_retry_reference_barrier_q_retry"],
          "Numerics.ale.driver_retry_reference_barrier_q_retry");
    }
    if (has_key(ale, "remap_damage_gate_enabled")) {
      numerics.ale.remap_damage_gate_enabled = strict_bool(
          ale["remap_damage_gate_enabled"],
          "Numerics.ale.remap_damage_gate_enabled");
    }
    if (has_key(ale, "remap_damage_dmax")) {
      numerics.ale.remap_damage_dmax =
          numeric_as_double(ale["remap_damage_dmax"], "Numerics.ale.remap_damage_dmax");
    }
    if (has_key(ale, "remap_damage_axis_eta")) {
      numerics.ale.remap_damage_axis_eta = numeric_as_double(
          ale["remap_damage_axis_eta"], "Numerics.ale.remap_damage_axis_eta");
    }
    if (has_key(ale, "remap_damage_axis_budget_enabled")) {
      numerics.ale.remap_damage_axis_budget_enabled = strict_bool(
          ale["remap_damage_axis_budget_enabled"],
          "Numerics.ale.remap_damage_axis_budget_enabled");
    }
    if (has_key(ale, "remap_damage_axis_budget_factor")) {
      numerics.ale.remap_damage_axis_budget_factor = numeric_as_double(
          ale["remap_damage_axis_budget_factor"],
          "Numerics.ale.remap_damage_axis_budget_factor");
    }
    if (has_key(ale, "predictive_acceptance_enabled")) {
      numerics.ale.predictive_acceptance_enabled = strict_bool(
          ale["predictive_acceptance_enabled"],
          "Numerics.ale.predictive_acceptance_enabled");
    }
    if (has_key(ale, "predictive_acceptance_axis_floor_fraction")) {
      numerics.ale.predictive_acceptance_axis_floor_fraction = numeric_as_double(
          ale["predictive_acceptance_axis_floor_fraction"],
          "Numerics.ale.predictive_acceptance_axis_floor_fraction");
    }
    if (has_key(ale, "predictive_acceptance_cell_vol_floor_fraction")) {
      numerics.ale.predictive_acceptance_cell_vol_floor_fraction = numeric_as_double(
          ale["predictive_acceptance_cell_vol_floor_fraction"],
          "Numerics.ale.predictive_acceptance_cell_vol_floor_fraction");
    }
    if (has_key(ale, "safe_backtrack_enabled")) {
      numerics.ale.safe_backtrack_enabled = strict_bool(
          ale["safe_backtrack_enabled"],
          "Numerics.ale.safe_backtrack_enabled");
    }
    if (has_key(ale, "safe_backtrack_min_exp")) {
      numerics.ale.safe_backtrack_min_exp = strict_int32(
          ale["safe_backtrack_min_exp"],
          "Numerics.ale.safe_backtrack_min_exp");
    }
    if (has_key(ale, "safe_backtrack_binary_iters")) {
      numerics.ale.safe_backtrack_binary_iters = strict_int32(
          ale["safe_backtrack_binary_iters"],
          "Numerics.ale.safe_backtrack_binary_iters");
    }
    if (has_key(ale, "mesh_epoch_enabled")) {
      numerics.ale.mesh_epoch_enabled = strict_bool(
          ale["mesh_epoch_enabled"],
          "Numerics.ale.mesh_epoch_enabled");
    }
    if (has_key(ale, "mesh_epoch_max_per_step")) {
      numerics.ale.mesh_epoch_max_per_step = strict_int32(
          ale["mesh_epoch_max_per_step"],
          "Numerics.ale.mesh_epoch_max_per_step");
      if (numerics.ale.mesh_epoch_max_per_step < 1) {
        throw ConfigError(
            "Numerics.ale.mesh_epoch_max_per_step must be >= 1");
      }
    }
    if (has_key(ale, "corner_cell_aspect_protection_enabled")) {
      numerics.ale.corner_cell_aspect_protection_enabled = strict_bool(
          ale["corner_cell_aspect_protection_enabled"],
          "Numerics.ale.corner_cell_aspect_protection_enabled");
    }
    if (has_key(ale, "corner_cell_aspect_eta")) {
      numerics.ale.corner_cell_aspect_eta = numeric_as_double(
          ale["corner_cell_aspect_eta"],
          "Numerics.ale.corner_cell_aspect_eta");
    }
    if (has_key(ale, "rezone_solver")) {
      numerics.ale.rezone_solver =
          strict_string(ale["rezone_solver"], "Numerics.ale.rezone_solver");
      if (numerics.ale.rezone_solver != "legacy_winslow" &&
          numerics.ale.rezone_solver != "rz_full_metric_winslow" &&
          numerics.ale.rezone_solver != "m1_tmop") {
        throw ValueError(
            "Numerics.ale.rezone_solver must be one of "
            "{\"legacy_winslow\", \"rz_full_metric_winslow\", \"m1_tmop\"}");
      }
    }
    if (has_key(ale, "m1_gamma_align")) {
      numerics.ale.m1_gamma_align = numeric_as_double(
          ale["m1_gamma_align"], "Numerics.ale.m1_gamma_align");
    }
    if (has_key(ale, "m1_lambda_tether")) {
      numerics.ale.m1_lambda_tether = numeric_as_double(
          ale["m1_lambda_tether"], "Numerics.ale.m1_lambda_tether");
    }
    if (has_key(ale, "m1_theta_reg")) {
      numerics.ale.m1_theta_reg = numeric_as_double(
          ale["m1_theta_reg"], "Numerics.ale.m1_theta_reg");
    }
    if (has_key(ale, "m1_sweeps")) {
      numerics.ale.m1_sweeps =
          strict_int32(ale["m1_sweeps"], "Numerics.ale.m1_sweeps");
    }
    if (has_key(ale, "m1_min_j_dec_rel")) {
      numerics.ale.m1_min_j_dec_rel = numeric_as_double(
          ale["m1_min_j_dec_rel"], "Numerics.ale.m1_min_j_dec_rel");
    }
    if (has_key(ale, "m1_barrier_beta")) {
      numerics.ale.m1_barrier_beta = numeric_as_double(
          ale["m1_barrier_beta"], "Numerics.ale.m1_barrier_beta");
    }
    if (has_key(ale, "euler_window")) {
      const py::handle euler_window_obj = ale["euler_window"];
      if (!py::isinstance<py::dict>(euler_window_obj)) {
        throw_value_type_error(
            "Numerics.ale.euler_window", "dict", euler_window_obj);
      }
      const py::dict euler_window =
          py::reinterpret_borrow<py::dict>(euler_window_obj);
      parse_euler_window_dict(
          euler_window,
          "Numerics.ale.euler_window",
          numerics.ale.euler_window);
      numerics.ale.euler_window.replay_table_path =
          resolve_namelist_relative_path(
              config, numerics.ale.euler_window.replay_table_path);
    }
    if (has_key(ale, "euler_windows")) {
      const py::handle euler_windows_obj = ale["euler_windows"];
      if (!py::isinstance<py::sequence>(euler_windows_obj) ||
          py::isinstance<py::str>(euler_windows_obj)) {
        throw_value_type_error(
            "Numerics.ale.euler_windows", "list[dict]", euler_windows_obj);
      }
      const py::sequence euler_windows =
          py::reinterpret_borrow<py::sequence>(euler_windows_obj);
      numerics.ale.euler_windows.clear();
      numerics.ale.euler_windows.reserve(euler_windows.size());
      for (std::size_t i = 0; i < euler_windows.size(); ++i) {
        const std::string path =
            "Numerics.ale.euler_windows[" + std::to_string(i) + "]";
        const py::handle euler_window_obj = euler_windows[i];
        if (!py::isinstance<py::dict>(euler_window_obj)) {
          throw_value_type_error(path, "dict", euler_window_obj);
        }
        const py::dict euler_window =
            py::reinterpret_borrow<py::dict>(euler_window_obj);
        Config::NumericsConfig::AleConfig::EulerWindowConfig config;
        parse_euler_window_dict(euler_window, path, config);
        config.replay_table_path =
            resolve_namelist_relative_path(this->config,
                                           config.replay_table_path);
        numerics.ale.euler_windows.push_back(std::move(config));
      }
    }
    if (has_key(ale, "band_ale")) {
      const py::handle band_ale_obj = ale["band_ale"];
      if (!py::isinstance<py::dict>(band_ale_obj)) {
        throw_value_type_error(
            "Numerics.ale.band_ale", "dict", band_ale_obj);
      }
      const py::dict band_ale =
          py::reinterpret_borrow<py::dict>(band_ale_obj);
      parse_band_ale_dict(
          band_ale, "Numerics.ale.band_ale", numerics.ale.band_ale);
    }
    if (has_key(ale, "evacuated_cell")) {
      const py::handle evacuated_cell_obj = ale["evacuated_cell"];
      if (!py::isinstance<py::dict>(evacuated_cell_obj)) {
        throw_value_type_error(
            "Numerics.ale.evacuated_cell", "dict", evacuated_cell_obj);
      }
      const py::dict evacuated_cell =
          py::reinterpret_borrow<py::dict>(evacuated_cell_obj);
      parse_evacuated_cell_dict(evacuated_cell,
                                "Numerics.ale.evacuated_cell",
                                numerics.ale.evacuated_cell);
    }
    if (has_key(ale, "rezone_local_admissibility_linesearch")) {
      numerics.ale.rezone_local_admissibility_linesearch = strict_bool(
          ale["rezone_local_admissibility_linesearch"],
          "Numerics.ale.rezone_local_admissibility_linesearch");
    }
    if (has_key(ale, "rezone_local_j_floor_rel")) {
      numerics.ale.rezone_local_j_floor_rel = numeric_as_double(
          ale["rezone_local_j_floor_rel"],
          "Numerics.ale.rezone_local_j_floor_rel");
    }
    if (has_key(ale, "rezone_local_linesearch_max_halves")) {
      numerics.ale.rezone_local_linesearch_max_halves = strict_int32(
          ale["rezone_local_linesearch_max_halves"],
          "Numerics.ale.rezone_local_linesearch_max_halves");
    }
    if (has_key(ale, "reject_zero_gauss_j")) {
      numerics.ale.reject_zero_gauss_j = strict_bool(
          ale["reject_zero_gauss_j"], "Numerics.ale.reject_zero_gauss_j");
    }
    if (has_key(ale, "zero_gauss_j_floor_rel")) {
      numerics.ale.zero_gauss_j_floor_rel = numeric_as_double(
          ale["zero_gauss_j_floor_rel"], "Numerics.ale.zero_gauss_j_floor_rel");
    }
    if (has_key(ale, "lambda_sweep_diagnostic_enabled")) {
      numerics.ale.lambda_sweep_diagnostic_enabled = strict_bool(
          ale["lambda_sweep_diagnostic_enabled"],
          "Numerics.ale.lambda_sweep_diagnostic_enabled");
    }
    if (has_key(ale, "lambda_sweep_target_cell_c")) {
      numerics.ale.lambda_sweep_target_cell_c = strict_int32(
          ale["lambda_sweep_target_cell_c"],
          "Numerics.ale.lambda_sweep_target_cell_c");
    }
    if (has_key(ale, "lambda_sweep_target_cell_i")) {
      numerics.ale.lambda_sweep_target_cell_i = strict_int32(
          ale["lambda_sweep_target_cell_i"],
          "Numerics.ale.lambda_sweep_target_cell_i");
    }
    if (has_key(ale, "lambda_sweep_target_cell_j")) {
      numerics.ale.lambda_sweep_target_cell_j = strict_int32(
          ale["lambda_sweep_target_cell_j"],
          "Numerics.ale.lambda_sweep_target_cell_j");
    }
    if (has_key(ale, "lambda_sweep_max_exp")) {
      numerics.ale.lambda_sweep_max_exp = strict_int32(
          ale["lambda_sweep_max_exp"],
          "Numerics.ale.lambda_sweep_max_exp");
    }
    if (has_key(ale, "corner_jacobian_post_tangle_enabled")) {
      numerics.ale.corner_jacobian_post_tangle_enabled = strict_bool(
          ale["corner_jacobian_post_tangle_enabled"],
          "Numerics.ale.corner_jacobian_post_tangle_enabled");
    }
    if (has_key(ale, "corner_post_tangle_strict_floor_enabled")) {
      numerics.ale.corner_post_tangle_strict_floor_enabled = strict_bool(
          ale["corner_post_tangle_strict_floor_enabled"],
          "Numerics.ale.corner_post_tangle_strict_floor_enabled");
    }
    if (has_key(ale, "local_boundary_repair_enabled")) {
      numerics.ale.local_boundary_repair_enabled = strict_bool(
          ale["local_boundary_repair_enabled"],
          "Numerics.ale.local_boundary_repair_enabled");
    }
    if (has_key(ale, "multi_node_boundary_repair_enabled")) {
      numerics.ale.multi_node_boundary_repair_enabled = strict_bool(
          ale["multi_node_boundary_repair_enabled"],
          "Numerics.ale.multi_node_boundary_repair_enabled");
    }
    if (has_key(ale, "multi_node_interior_repair_enabled")) {
      numerics.ale.multi_node_interior_repair_enabled = strict_bool(
          ale["multi_node_interior_repair_enabled"],
          "Numerics.ale.multi_node_interior_repair_enabled");
    }
    if (has_key(ale, "axis_variational_projection_enabled")) {
      numerics.ale.axis_variational_projection_enabled = strict_bool(
          ale["axis_variational_projection_enabled"],
          "Numerics.ale.axis_variational_projection_enabled");
    }
    if (has_key(ale, "emergency_cell_deactivation_enabled")) {
      numerics.ale.emergency_cell_deactivation_enabled = strict_bool(
          ale["emergency_cell_deactivation_enabled"],
          "Numerics.ale.emergency_cell_deactivation_enabled");
    }
    if (has_key(ale, "multiblock_cross_seam_rezone_enabled")) {
      numerics.ale.multiblock_cross_seam_rezone_enabled = strict_bool(
          ale["multiblock_cross_seam_rezone_enabled"],
          "Numerics.ale.multiblock_cross_seam_rezone_enabled");
    }
    if (has_key(ale, "multiblock_scaled_reference_enabled")) {
      numerics.ale.multiblock_scaled_reference_enabled = strict_bool(
          ale["multiblock_scaled_reference_enabled"],
          "Numerics.ale.multiblock_scaled_reference_enabled");
    }
    if (has_key(ale, "multiblock_differential_reference_enabled")) {
      numerics.ale.multiblock_differential_reference_enabled = strict_bool(
          ale["multiblock_differential_reference_enabled"],
          "Numerics.ale.multiblock_differential_reference_enabled");
    }
    if (has_key(ale, "multiblock_differential_reference_band_count")) {
      numerics.ale.multiblock_differential_reference_band_count = strict_int32(
          ale["multiblock_differential_reference_band_count"],
          "Numerics.ale.multiblock_differential_reference_band_count");
    }
    if (has_key(ale, "multiblock_differential_reference_smoothing_g0")) {
      numerics.ale.multiblock_differential_reference_smoothing_g0 =
          numeric_as_double(
              ale["multiblock_differential_reference_smoothing_g0"],
              "Numerics.ale.multiblock_differential_reference_smoothing_g0");
    }
    if (has_key(ale, "multiblock_differential_reference_nu")) {
      numerics.ale.multiblock_differential_reference_nu = numeric_as_double(
          ale["multiblock_differential_reference_nu"],
          "Numerics.ale.multiblock_differential_reference_nu");
    }
    if (has_key(ale, "multiblock_differential_reference_eps_v")) {
      numerics.ale.multiblock_differential_reference_eps_v = numeric_as_double(
          ale["multiblock_differential_reference_eps_v"],
          "Numerics.ale.multiblock_differential_reference_eps_v");
    }
    if (has_key(ale, "multiblock_differential_reference_s_cap_min_rel")) {
      numerics.ale.multiblock_differential_reference_s_cap_min_rel =
          numeric_as_double(
              ale["multiblock_differential_reference_s_cap_min_rel"],
              "Numerics.ale.multiblock_differential_reference_s_cap_min_rel");
    }
    if (has_key(ale, "multiblock_differential_reference_xi_seam_tol")) {
      numerics.ale.multiblock_differential_reference_xi_seam_tol =
          numeric_as_double(
              ale["multiblock_differential_reference_xi_seam_tol"],
              "Numerics.ale.multiblock_differential_reference_xi_seam_tol");
    }
    if (has_key(ale, "multiblock_differential_reference_sigma_warn_floor")) {
      numerics.ale.multiblock_differential_reference_sigma_warn_floor =
          numeric_as_double(
              ale["multiblock_differential_reference_sigma_warn_floor"],
              "Numerics.ale.multiblock_differential_reference_sigma_warn_floor");
    }
    if (has_key(ale, "multiblock_lagrangian_bulk_center_patch_reference_enabled")) {
      numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled =
          strict_bool(
              ale["multiblock_lagrangian_bulk_center_patch_reference_enabled"],
              "Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled");
    }
    if (has_key(ale, "multiblock_center_patch_ring_max")) {
      numerics.ale.multiblock_center_patch_ring_max = strict_int32(
          ale["multiblock_center_patch_ring_max"],
          "Numerics.ale.multiblock_center_patch_ring_max");
    }
    if (has_key(ale, "multiblock_center_patch_xi_center")) {
      numerics.ale.multiblock_center_patch_xi_center = numeric_as_double(
          ale["multiblock_center_patch_xi_center"],
          "Numerics.ale.multiblock_center_patch_xi_center");
    }
    if (has_key(ale, "multiblock_center_patch_halo_layers")) {
      numerics.ale.multiblock_center_patch_halo_layers = strict_int32(
          ale["multiblock_center_patch_halo_layers"],
          "Numerics.ale.multiblock_center_patch_halo_layers");
    }
    if (has_key(ale, "multiblock_center_patch_vol_on")) {
      numerics.ale.multiblock_center_patch_vol_on = numeric_as_double(
          ale["multiblock_center_patch_vol_on"],
          "Numerics.ale.multiblock_center_patch_vol_on");
    }
    if (has_key(ale, "multiblock_center_patch_vol_off")) {
      numerics.ale.multiblock_center_patch_vol_off = numeric_as_double(
          ale["multiblock_center_patch_vol_off"],
          "Numerics.ale.multiblock_center_patch_vol_off");
    }
    if (has_key(ale, "multiblock_center_patch_cornerj_on")) {
      numerics.ale.multiblock_center_patch_cornerj_on = numeric_as_double(
          ale["multiblock_center_patch_cornerj_on"],
          "Numerics.ale.multiblock_center_patch_cornerj_on");
    }
    if (has_key(ale, "multiblock_center_patch_cornerj_off")) {
      numerics.ale.multiblock_center_patch_cornerj_off = numeric_as_double(
          ale["multiblock_center_patch_cornerj_off"],
          "Numerics.ale.multiblock_center_patch_cornerj_off");
    }
    if (has_key(ale, "multiblock_center_patch_gaussj_on")) {
      numerics.ale.multiblock_center_patch_gaussj_on = numeric_as_double(
          ale["multiblock_center_patch_gaussj_on"],
          "Numerics.ale.multiblock_center_patch_gaussj_on");
    }
    if (has_key(ale, "multiblock_center_patch_gaussj_off")) {
      numerics.ale.multiblock_center_patch_gaussj_off = numeric_as_double(
          ale["multiblock_center_patch_gaussj_off"],
          "Numerics.ale.multiblock_center_patch_gaussj_off");
    }
    if (has_key(ale, "ale_reference_diagnostics_enabled")) {
      numerics.ale.ale_reference_diagnostics_enabled = strict_bool(
          ale["ale_reference_diagnostics_enabled"],
          "Numerics.ale.ale_reference_diagnostics_enabled");
    }
    if (has_key(ale, "multiblock_path_admissibility_enabled")) {
      numerics.ale.multiblock_path_admissibility_enabled = strict_bool(
          ale["multiblock_path_admissibility_enabled"],
          "Numerics.ale.multiblock_path_admissibility_enabled");
    }
    if (has_key(ale, "path_admissibility_floor")) {
      numerics.ale.path_admissibility_floor = numeric_as_double(
          ale["path_admissibility_floor"],
          "Numerics.ale.path_admissibility_floor");
    }
    if (has_key(ale, "dt_rejection_factor")) {
      numerics.ale.dt_rejection_factor = numeric_as_double(
          ale["dt_rejection_factor"], "Numerics.ale.dt_rejection_factor");
    }
    if (has_key(ale, "max_dt_rejections")) {
      numerics.ale.max_dt_rejections = strict_int32(
          ale["max_dt_rejections"], "Numerics.ale.max_dt_rejections");
    }
    if (has_key(ale, "axis_band_managed_remap_enabled")) {
      numerics.ale.axis_band_managed_remap_enabled = strict_bool(
          ale["axis_band_managed_remap_enabled"],
          "Numerics.ale.axis_band_managed_remap_enabled");
    }
    if (has_key(ale, "axis_band_managed_remap_width")) {
      numerics.ale.axis_band_managed_remap_width = strict_int32(
          ale["axis_band_managed_remap_width"],
          "Numerics.ale.axis_band_managed_remap_width");
    }
    if (has_key(ale, "axis_band_managed_remap_max_width")) {
      numerics.ale.axis_band_managed_remap_max_width = strict_int32(
          ale["axis_band_managed_remap_max_width"],
          "Numerics.ale.axis_band_managed_remap_max_width");
    }
    if (has_key(ale, "axis_band_managed_remap_every_hydro_half_step")) {
      numerics.ale.axis_band_managed_remap_every_hydro_half_step = strict_bool(
          ale["axis_band_managed_remap_every_hydro_half_step"],
          "Numerics.ale.axis_band_managed_remap_every_hydro_half_step");
    }
    if (has_key(ale, "axis_band_managed_remap_margin_trigger")) {
      numerics.ale.axis_band_managed_remap_margin_trigger = numeric_as_double(
          ale["axis_band_managed_remap_margin_trigger"],
          "Numerics.ale.axis_band_managed_remap_margin_trigger");
    }
    if (has_key(ale, "axis_band_managed_remap_equal_volume")) {
      numerics.ale.axis_band_managed_remap_equal_volume = strict_bool(
          ale["axis_band_managed_remap_equal_volume"],
          "Numerics.ale.axis_band_managed_remap_equal_volume");
    }
    if (has_key(ale, "axis_band_managed_remap_include_radiation_groups")) {
      numerics.ale.axis_band_managed_remap_include_radiation_groups = strict_bool(
          ale["axis_band_managed_remap_include_radiation_groups"],
          "Numerics.ale.axis_band_managed_remap_include_radiation_groups");
    }
    if (has_key(ale, "axis_rezone_enabled")) {
      numerics.ale.axis_rezone_enabled = strict_bool(
          ale["axis_rezone_enabled"],
          "Numerics.ale.axis_rezone_enabled");
    }
    if (has_key(ale, "axis_rezone_trigger_edge_fraction")) {
      numerics.ale.axis_rezone_trigger_edge_fraction = numeric_as_double(
          ale["axis_rezone_trigger_edge_fraction"],
          "Numerics.ale.axis_rezone_trigger_edge_fraction");
    }
    if (has_key(ale, "axis_rezone_trigger_min_altitude_fraction")) {
      numerics.ale.axis_rezone_trigger_min_altitude_fraction = numeric_as_double(
          ale["axis_rezone_trigger_min_altitude_fraction"],
          "Numerics.ale.axis_rezone_trigger_min_altitude_fraction");
    }
    if (has_key(ale, "axis_rezone_eta_floor")) {
      numerics.ale.axis_rezone_eta_floor = numeric_as_double(
          ale["axis_rezone_eta_floor"],
          "Numerics.ale.axis_rezone_eta_floor");
    }
    if (has_key(ale, "core_freeze_enabled")) {
      numerics.ale.core_freeze_enabled = strict_bool(
          ale["core_freeze_enabled"],
          "Numerics.ale.core_freeze_enabled");
    }
    if (has_key(ale, "core_freeze_source")) {
      numerics.ale.core_freeze_source = strict_string(
          ale["core_freeze_source"],
          "Numerics.ale.core_freeze_source");
      if (numerics.ale.core_freeze_source != "gas_tracer") {
        throw ValueError(
            "Numerics.ale.core_freeze_source must be \"gas_tracer\" in S1");
      }
    }
    if (has_key(ale, "core_freeze_tracer_cut")) {
      numerics.ale.core_freeze_tracer_cut = numeric_as_double(
          ale["core_freeze_tracer_cut"],
          "Numerics.ale.core_freeze_tracer_cut");
    }
    if (has_key(ale, "core_freeze_halo_layers")) {
      numerics.ale.core_freeze_halo_layers = strict_int32(
          ale["core_freeze_halo_layers"],
          "Numerics.ale.core_freeze_halo_layers");
    }
    if (has_key(ale, "core_freeze_apply_to_axis_rezone")) {
      numerics.ale.core_freeze_apply_to_axis_rezone = strict_bool(
          ale["core_freeze_apply_to_axis_rezone"],
          "Numerics.ale.core_freeze_apply_to_axis_rezone");
    }
    if (has_key(ale, "core_freeze_skip_velocity_projection")) {
      numerics.ale.core_freeze_skip_velocity_projection = strict_bool(
          ale["core_freeze_skip_velocity_projection"],
          "Numerics.ale.core_freeze_skip_velocity_projection");
    }
    if (numerics.ale.axis_band_managed_remap_width < 1) {
      throw ValueError("Numerics.ale.axis_band_managed_remap_width must be >= 1");
    }
    if (numerics.ale.axis_band_managed_remap_max_width <
        numerics.ale.axis_band_managed_remap_width) {
      throw ValueError(
          "Numerics.ale.axis_band_managed_remap_max_width must be >= "
          "axis_band_managed_remap_width");
    }
    if (numerics.ale.axis_band_managed_remap_max_width > 32) {
      throw ValueError(
          "Numerics.ale.axis_band_managed_remap_max_width must be <= 32");
    }
    if (!(numerics.ale.path_admissibility_floor > 0.0)) {
      throw ValueError("Numerics.ale.path_admissibility_floor must be > 0");
    }
    if (numerics.ale.multiblock_differential_reference_band_count < 8 ||
        numerics.ale.multiblock_differential_reference_band_count > 4096) {
      throw ValueError(
          "Numerics.ale.multiblock_differential_reference_band_count must be in [8, 4096]");
    }
    if (!(numerics.ale.multiblock_differential_reference_smoothing_g0 > 0.0 &&
          numerics.ale.multiblock_differential_reference_smoothing_g0 <= 1.0)) {
      throw ValueError(
          "Numerics.ale.multiblock_differential_reference_smoothing_g0 must be in (0, 1]");
    }
    if (!(numerics.ale.multiblock_differential_reference_nu > 0.0 &&
          numerics.ale.multiblock_differential_reference_nu <= 1.0)) {
      throw ValueError(
          "Numerics.ale.multiblock_differential_reference_nu must be in (0, 1]");
    }
    if (!(numerics.ale.multiblock_differential_reference_eps_v > 0.0 &&
          numerics.ale.multiblock_differential_reference_eps_v <= 1.0)) {
      throw ValueError(
          "Numerics.ale.multiblock_differential_reference_eps_v must be in (0, 1]");
    }
    if (!(numerics.ale.multiblock_differential_reference_s_cap_min_rel > 0.0 &&
          numerics.ale.multiblock_differential_reference_s_cap_min_rel <= 1.0)) {
      throw ValueError(
          "Numerics.ale.multiblock_differential_reference_s_cap_min_rel must be in (0, 1]");
    }
    if (!(numerics.ale.multiblock_differential_reference_xi_seam_tol > 0.0 &&
          numerics.ale.multiblock_differential_reference_xi_seam_tol <= 1.0e-3)) {
      throw ValueError(
          "Numerics.ale.multiblock_differential_reference_xi_seam_tol must be in (0, 1e-3]");
    }
    if (!(numerics.ale.multiblock_differential_reference_sigma_warn_floor > 0.0 &&
          numerics.ale.multiblock_differential_reference_sigma_warn_floor <= 1.0)) {
      throw ValueError(
          "Numerics.ale.multiblock_differential_reference_sigma_warn_floor must be in (0, 1]");
    }
    if (!(numerics.ale.dt_rejection_factor > 0.0 &&
          numerics.ale.dt_rejection_factor < 1.0)) {
      throw ValueError("Numerics.ale.dt_rejection_factor must be in (0, 1)");
    }
    if (numerics.ale.max_dt_rejections < 1) {
      throw ValueError("Numerics.ale.max_dt_rejections must be >= 1");
    }
    if (!(numerics.ale.axis_band_managed_remap_margin_trigger > 0.0)) {
      throw ValueError(
          "Numerics.ale.axis_band_managed_remap_margin_trigger must be > 0");
    }
    if (!(numerics.ale.axis_rezone_trigger_edge_fraction > 0.0 &&
          numerics.ale.axis_rezone_trigger_edge_fraction <= 1.0)) {
      throw ValueError(
          "Numerics.ale.axis_rezone_trigger_edge_fraction must be in (0, 1]");
    }
    if (!(numerics.ale.axis_rezone_trigger_min_altitude_fraction > 0.0 &&
          numerics.ale.axis_rezone_trigger_min_altitude_fraction <= 1.0)) {
      throw ValueError(
          "Numerics.ale.axis_rezone_trigger_min_altitude_fraction must be in (0, 1]");
    }
    if (!(numerics.ale.axis_rezone_eta_floor > 0.0 &&
          numerics.ale.axis_rezone_eta_floor < 1.0)) {
      throw ValueError("Numerics.ale.axis_rezone_eta_floor must be in (0, 1)");
    }
    if (numerics.ale.core_freeze_source != "gas_tracer") {
      throw ValueError(
          "Numerics.ale.core_freeze_source must be \"gas_tracer\" in S1");
    }
    if (!(numerics.ale.core_freeze_tracer_cut >= 0.0 &&
          numerics.ale.core_freeze_tracer_cut <= 1.0)) {
      throw ValueError(
          "Numerics.ale.core_freeze_tracer_cut must be in [0, 1]");
    }
    if (numerics.ale.core_freeze_halo_layers < 0) {
      throw ValueError("Numerics.ale.core_freeze_halo_layers must be >= 0");
    }
    if (numerics.ale.safe_backtrack_min_exp < 0 ||
        numerics.ale.safe_backtrack_min_exp > 60) {
      throw ValueError("Numerics.ale.safe_backtrack_min_exp must be in [0, 60]");
    }
    if (numerics.ale.safe_backtrack_binary_iters < 0 ||
        numerics.ale.safe_backtrack_binary_iters > 60) {
      throw ValueError("Numerics.ale.safe_backtrack_binary_iters must be in [0, 60]");
    }
    if (!(numerics.ale.corner_cell_aspect_eta >= 0.0 &&
          numerics.ale.corner_cell_aspect_eta <= 1.0)) {
      throw ValueError("Numerics.ale.corner_cell_aspect_eta must be in [0, 1]");
    }
    if (numerics.ale.rezone_solver != "legacy_winslow" &&
        numerics.ale.rezone_solver != "rz_full_metric_winslow" &&
        numerics.ale.rezone_solver != "m1_tmop") {
      throw ValueError(
          "Numerics.ale.rezone_solver must be one of "
          "{\"legacy_winslow\", \"rz_full_metric_winslow\", \"m1_tmop\"}");
    }
    if (!(numerics.ale.m1_gamma_align >= 0.0)) {
      throw ValueError("Numerics.ale.m1_gamma_align must be >= 0");
    }
    if (!(numerics.ale.m1_lambda_tether >= 0.0)) {
      throw ValueError("Numerics.ale.m1_lambda_tether must be >= 0");
    }
    if (!(numerics.ale.m1_theta_reg >= 0.0)) {
      throw ValueError("Numerics.ale.m1_theta_reg must be >= 0");
    }
    if (numerics.ale.m1_sweeps < 1) {
      throw ValueError("Numerics.ale.m1_sweeps must be >= 1");
    }
    if (!std::isfinite(numerics.ale.m1_min_j_dec_rel) ||
        numerics.ale.m1_min_j_dec_rel < 0.0) {
      throw ValueError(
          "Numerics.ale.m1_min_j_dec_rel must be finite and >= 0");
    }
    if (!(numerics.ale.m1_barrier_beta >= 0.0)) {
      throw ValueError("Numerics.ale.m1_barrier_beta must be >= 0");
    }
    validate_euler_window_config(
        numerics.ale.euler_window, "Numerics.ale.euler_window");
    for (std::size_t i = 0; i < numerics.ale.euler_windows.size(); ++i) {
      validate_euler_window_config(
          numerics.ale.euler_windows[i],
          "Numerics.ale.euler_windows[" + std::to_string(i) + "]");
    }
    validate_band_ale_config(
        numerics.ale.band_ale, "Numerics.ale.band_ale");
    validate_evacuated_cell_config(
        numerics.ale.evacuated_cell, "Numerics.ale.evacuated_cell");
    if (!(numerics.ale.rezone_local_j_floor_rel >= 0.0)) {
      throw ValueError("Numerics.ale.rezone_local_j_floor_rel must be >= 0");
    }
    if (numerics.ale.rezone_local_linesearch_max_halves < 0 ||
        numerics.ale.rezone_local_linesearch_max_halves > 32) {
      throw ValueError(
          "Numerics.ale.rezone_local_linesearch_max_halves must be in [0, 32]");
    }
    if (!(numerics.ale.zero_gauss_j_floor_rel > 0.0)) {
      throw ValueError("Numerics.ale.zero_gauss_j_floor_rel must be > 0");
    }
    if (numerics.ale.lambda_sweep_target_cell_c < -1) {
      throw ValueError("Numerics.ale.lambda_sweep_target_cell_c must be >= -1");
    }
    if (numerics.ale.lambda_sweep_target_cell_i < -1) {
      throw ValueError("Numerics.ale.lambda_sweep_target_cell_i must be >= -1");
    }
    if (numerics.ale.lambda_sweep_target_cell_j < -1) {
      throw ValueError("Numerics.ale.lambda_sweep_target_cell_j must be >= -1");
    }
    if ((numerics.ale.lambda_sweep_target_cell_i >= 0) !=
        (numerics.ale.lambda_sweep_target_cell_j >= 0)) {
      throw ValueError(
          "Numerics.ale.lambda_sweep_target_cell_i and "
          "lambda_sweep_target_cell_j must both be set or both be -1");
    }
    if (numerics.ale.lambda_sweep_max_exp < 0 ||
        numerics.ale.lambda_sweep_max_exp > 1022) {
      throw ValueError("Numerics.ale.lambda_sweep_max_exp must be in [0, 1022]");
    }
    if (has_key(ale, "axis_repair_mode")) {
      numerics.ale.axis_repair_mode =
          strict_string(ale["axis_repair_mode"], "Numerics.ale.axis_repair_mode");
      if (numerics.ale.axis_repair_mode != "full_winslow" &&
          numerics.ale.axis_repair_mode != "axis_spine_only" &&
          numerics.ale.axis_repair_mode != "axis_z_winslow" &&
          numerics.ale.axis_repair_mode != "none") {
        throw ValueError(
            "Numerics.ale.axis_repair_mode must be one of "
            "{\"full_winslow\", \"axis_spine_only\", \"axis_z_winslow\", \"none\"}");
      }
    }
    if (has_key(ale, "remap_scheme")) {
      numerics.ale.remap_scheme =
          strict_string(ale["remap_scheme"], "Numerics.ale.remap_scheme");
      if (numerics.ale.remap_scheme != "legacy_split" &&
          numerics.ale.remap_scheme != "ms2_moments") {
        throw ValueError(
            "Numerics.ale.remap_scheme must be one of "
            "{\"legacy_split\", \"ms2_moments\"}");
      }
    }
    if (has_key(ale, "remap_ms2_limiter")) {
      numerics.ale.remap_ms2_limiter =
          strict_string(ale["remap_ms2_limiter"], "Numerics.ale.remap_ms2_limiter");
      if (numerics.ale.remap_ms2_limiter != "van_leer" &&
          numerics.ale.remap_ms2_limiter != "barth_jespersen") {
        throw ValueError(
            "Numerics.ale.remap_ms2_limiter must be one of "
            "{\"van_leer\", \"barth_jespersen\"}");
      }
    }
    if (has_key(ale, "conservative_remap_enabled")) {
      numerics.ale.conservative_remap_enabled = strict_bool(
          ale["conservative_remap_enabled"],
          "Numerics.ale.conservative_remap_enabled");
    }
    if (has_key(ale, "conservative_remap_target")) {
      numerics.ale.conservative_remap_target = strict_string(
          ale["conservative_remap_target"],
          "Numerics.ale.conservative_remap_target");
      if (numerics.ale.conservative_remap_target != "reference") {
        throw ValueError(
            "Numerics.ale.conservative_remap_target must be \"reference\"");
      }
    }
    if (has_key(ale, "conservative_remap_radiation_enabled")) {
      numerics.ale.conservative_remap_radiation_enabled = strict_bool(
          ale["conservative_remap_radiation_enabled"],
          "Numerics.ale.conservative_remap_radiation_enabled");
    }
    if (has_key(ale, "conservative_remap_order")) {
      numerics.ale.conservative_remap_order = strict_string(
          ale["conservative_remap_order"],
          "Numerics.ale.conservative_remap_order");
      if (numerics.ale.conservative_remap_order != "first_order_donor" &&
          numerics.ale.conservative_remap_order != "second_order_van_leer") {
        throw ValueError(
            "Numerics.ale.conservative_remap_order must be one of "
            "{\"first_order_donor\", \"second_order_van_leer\"}");
      }
    }
    if (has_key(ale, "tri_fan_tracking_reference_enabled")) {
      numerics.ale.tri_fan_tracking_reference_enabled = strict_bool(
          ale["tri_fan_tracking_reference_enabled"],
          "Numerics.ale.tri_fan_tracking_reference_enabled");
    }
    if (has_key(ale, "tri_fan_tracking_reference_mode")) {
      numerics.ale.tri_fan_tracking_reference_mode = strict_string(
          ale["tri_fan_tracking_reference_mode"],
          "Numerics.ale.tri_fan_tracking_reference_mode");
      if (numerics.ale.tri_fan_tracking_reference_mode != "legacy_lagging" &&
          numerics.ale.tri_fan_tracking_reference_mode != "seamless_converging") {
        throw ValueError(
            "Numerics.ale.tri_fan_tracking_reference_mode must be one of "
            "{\"legacy_lagging\", \"seamless_converging\"}");
      }
    }
    if (has_key(ale, "tri_fan_tracking_reference_omega")) {
      numerics.ale.tri_fan_tracking_reference_omega = numeric_as_double(
          ale["tri_fan_tracking_reference_omega"],
          "Numerics.ale.tri_fan_tracking_reference_omega");
    }
    if (has_key(ale, "tri_fan_tracking_reference_beta")) {
      numerics.ale.tri_fan_tracking_reference_beta = numeric_as_double(
          ale["tri_fan_tracking_reference_beta"],
          "Numerics.ale.tri_fan_tracking_reference_beta");
    }
    if (has_key(ale, "tri_fan_tracking_reference_g0")) {
      numerics.ale.tri_fan_tracking_reference_g0 = numeric_as_double(
          ale["tri_fan_tracking_reference_g0"],
          "Numerics.ale.tri_fan_tracking_reference_g0");
    }
    if (has_key(ale, "tri_fan_tracking_reference_nu")) {
      numerics.ale.tri_fan_tracking_reference_nu = numeric_as_double(
          ale["tri_fan_tracking_reference_nu"],
          "Numerics.ale.tri_fan_tracking_reference_nu");
    }
    if (has_key(ale, "tri_fan_tracking_reference_eps_v")) {
      numerics.ale.tri_fan_tracking_reference_eps_v = numeric_as_double(
          ale["tri_fan_tracking_reference_eps_v"],
          "Numerics.ale.tri_fan_tracking_reference_eps_v");
    }
    if (has_key(ale, "conservative_remap_lagrangian_bulk_enabled")) {
      numerics.ale.conservative_remap_lagrangian_bulk_enabled = strict_bool(
          ale["conservative_remap_lagrangian_bulk_enabled"],
          "Numerics.ale.conservative_remap_lagrangian_bulk_enabled");
    }
    if (has_key(ale, "conservative_remap_lagrangian_bulk_center_node_ring_max")) {
      numerics.ale.conservative_remap_lagrangian_bulk_center_node_ring_max =
          strict_int32(
              ale["conservative_remap_lagrangian_bulk_center_node_ring_max"],
              "Numerics.ale.conservative_remap_lagrangian_bulk_center_node_ring_max");
      if (numerics.ale.conservative_remap_lagrangian_bulk_center_node_ring_max < 0) {
        throw ValueError(
            "Numerics.ale.conservative_remap_lagrangian_bulk_center_node_ring_max "
            "must be >= 0");
      }
    }
    if (has_key(ale, "central_pseudo_core_enabled")) {
      numerics.ale.central_pseudo_core_enabled = strict_bool(
          ale["central_pseudo_core_enabled"],
          "Numerics.ale.central_pseudo_core_enabled");
    }
    if (has_key(ale, "central_pseudo_core_s_c")) {
      numerics.ale.central_pseudo_core_s_c = numeric_as_double(
          ale["central_pseudo_core_s_c"],
          "Numerics.ale.central_pseudo_core_s_c");
    }
    if (has_key(ale, "central_pseudo_core_activation_time_s")) {
      numerics.ale.central_pseudo_core_activation_time_s = numeric_as_double(
          ale["central_pseudo_core_activation_time_s"],
          "Numerics.ale.central_pseudo_core_activation_time_s");
      if (!(std::isfinite(
                numerics.ale.central_pseudo_core_activation_time_s) &&
            numerics.ale.central_pseudo_core_activation_time_s >= 0.0)) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_activation_time_s must be "
            "finite and >= 0");
      }
    }
    if (has_key(ale, "central_pseudo_core_ring_absorption_enabled")) {
      numerics.ale.central_pseudo_core_ring_absorption_enabled = strict_bool(
          ale["central_pseudo_core_ring_absorption_enabled"],
          "Numerics.ale.central_pseudo_core_ring_absorption_enabled");
    }
    if (has_key(ale, "central_pseudo_core_ring_absorption_tau")) {
      numerics.ale.central_pseudo_core_ring_absorption_tau = numeric_as_double(
          ale["central_pseudo_core_ring_absorption_tau"],
          "Numerics.ale.central_pseudo_core_ring_absorption_tau");
      if (!(numerics.ale.central_pseudo_core_ring_absorption_tau > 0.0 &&
            numerics.ale.central_pseudo_core_ring_absorption_tau < 1.0)) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_ring_absorption_tau must be in "
            "(0, 1)");
      }
    }
    if (has_key(ale, "conv_rezone_enabled")) {
      numerics.ale.conv_rezone_enabled = strict_bool(
          ale["conv_rezone_enabled"], "Numerics.ale.conv_rezone_enabled");
    }
    if (has_key(ale, "central_pseudo_core_core1d_enabled")) {
      numerics.ale.central_pseudo_core_core1d_enabled = strict_bool(
          ale["central_pseudo_core_core1d_enabled"],
          "Numerics.ale.central_pseudo_core_core1d_enabled");
    }
    if (has_key(ale, "central_pseudo_core_core1d_build_shells")) {
      numerics.ale.central_pseudo_core_core1d_build_shells =
          strict_int32(ale["central_pseudo_core_core1d_build_shells"],
                       "Numerics.ale.central_pseudo_core_core1d_build_shells");
      if (numerics.ale.central_pseudo_core_core1d_build_shells < 4 ||
          numerics.ale.central_pseudo_core_core1d_build_shells > 4096) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_core1d_build_shells must be in "
            "[4, 4096]");
      }
    }
    if (has_key(ale, "central_pseudo_core_core1d_split_append")) {
      numerics.ale.central_pseudo_core_core1d_split_append =
          strict_int32(ale["central_pseudo_core_core1d_split_append"],
                       "Numerics.ale.central_pseudo_core_core1d_split_append");
      if (numerics.ale.central_pseudo_core_core1d_split_append < 0 ||
          numerics.ale.central_pseudo_core_core1d_split_append > 1024) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_core1d_split_append must be in "
            "[0, 1024]");
      }
    }
    if (has_key(ale, "central_pseudo_core_core1d_av_c1")) {
      numerics.ale.central_pseudo_core_core1d_av_c1 = numeric_as_double(
          ale["central_pseudo_core_core1d_av_c1"],
          "Numerics.ale.central_pseudo_core_core1d_av_c1");
      if (!(numerics.ale.central_pseudo_core_core1d_av_c1 > 0.0)) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_core1d_av_c1 must be > 0");
      }
    }
    if (has_key(ale, "central_pseudo_core_core1d_av_c2")) {
      numerics.ale.central_pseudo_core_core1d_av_c2 = numeric_as_double(
          ale["central_pseudo_core_core1d_av_c2"],
          "Numerics.ale.central_pseudo_core_core1d_av_c2");
      if (!(numerics.ale.central_pseudo_core_core1d_av_c2 > 0.0)) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_core1d_av_c2 must be > 0");
      }
    }
    if (has_key(ale, "central_pseudo_core_core1d_cfl")) {
      numerics.ale.central_pseudo_core_core1d_cfl = numeric_as_double(
          ale["central_pseudo_core_core1d_cfl"],
          "Numerics.ale.central_pseudo_core_core1d_cfl");
      if (!(numerics.ale.central_pseudo_core_core1d_cfl > 0.0 &&
            numerics.ale.central_pseudo_core_core1d_cfl < 1.0)) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_core1d_cfl must be in (0, 1)");
      }
    }
    if (has_key(ale, "central_pseudo_core_core1d_piston_cap")) {
      numerics.ale.central_pseudo_core_core1d_piston_cap = numeric_as_double(
          ale["central_pseudo_core_core1d_piston_cap"],
          "Numerics.ale.central_pseudo_core_core1d_piston_cap");
      if (!(numerics.ale.central_pseudo_core_core1d_piston_cap > 0.0)) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_core1d_piston_cap must be > 0");
      }
    }
    if (has_key(ale, "central_pseudo_core_core1d_dist_append")) {
      numerics.ale.central_pseudo_core_core1d_dist_append = strict_bool(
          ale["central_pseudo_core_core1d_dist_append"],
          "Numerics.ale.central_pseudo_core_core1d_dist_append");
    }
    if (has_key(ale, "central_pseudo_core_core1d_max_substeps")) {
      numerics.ale.central_pseudo_core_core1d_max_substeps =
          strict_int32(ale["central_pseudo_core_core1d_max_substeps"],
                       "Numerics.ale.central_pseudo_core_core1d_max_substeps");
      if (numerics.ale.central_pseudo_core_core1d_max_substeps < 1) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_core1d_max_substeps must be >= 1");
      }
    }
    if (has_key(ale, "central_pseudo_core_spherical_absorb_gasfront")) {
      numerics.ale.central_pseudo_core_spherical_absorb_gasfront =
          strict_bool(ale["central_pseudo_core_spherical_absorb_gasfront"],
                      "Numerics.ale.central_pseudo_core_spherical_absorb_gasfront");
    }
    if (has_key(ale, "central_pseudo_core_spherical_absorb_alpha")) {
      numerics.ale.central_pseudo_core_spherical_absorb_alpha =
          numeric_as_double(
              ale["central_pseudo_core_spherical_absorb_alpha"],
              "Numerics.ale.central_pseudo_core_spherical_absorb_alpha");
      if (!(numerics.ale.central_pseudo_core_spherical_absorb_alpha == 0.0 ||
            (numerics.ale.central_pseudo_core_spherical_absorb_alpha > 0.0 &&
             numerics.ale.central_pseudo_core_spherical_absorb_alpha < 1.0))) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_spherical_absorb_alpha must be "
            "0 or in (0, 1)");
      }
    }
    if (has_key(ale, "central_pseudo_core_spherical_absorb_pjump")) {
      numerics.ale.central_pseudo_core_spherical_absorb_pjump =
          numeric_as_double(
              ale["central_pseudo_core_spherical_absorb_pjump"],
              "Numerics.ale.central_pseudo_core_spherical_absorb_pjump");
      if (!(numerics.ale.central_pseudo_core_spherical_absorb_pjump == 0.0 ||
            numerics.ale.central_pseudo_core_spherical_absorb_pjump > 1.0)) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_spherical_absorb_pjump must be "
            "0 or > 1");
      }
    }
    if (has_key(ale, "central_pseudo_core_mixed_absorb_enabled")) {
      numerics.ale.central_pseudo_core_mixed_absorb_enabled = strict_bool(
          ale["central_pseudo_core_mixed_absorb_enabled"],
          "Numerics.ale.central_pseudo_core_mixed_absorb_enabled");
    }
    if (has_key(ale, "central_pseudo_core_absorb_watch_rows")) {
      numerics.ale.central_pseudo_core_absorb_watch_rows = strict_int32(
          ale["central_pseudo_core_absorb_watch_rows"],
          "Numerics.ale.central_pseudo_core_absorb_watch_rows");
      if (numerics.ale.central_pseudo_core_absorb_watch_rows < 1 ||
          numerics.ale.central_pseudo_core_absorb_watch_rows > 8) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_absorb_watch_rows must be in "
            "[1, 8]");
      }
    }
    if (has_key(ale, "remap_mass_closure_reject_tol")) {
      numerics.ale.remap_mass_closure_reject_tol = numeric_as_double(
          ale["remap_mass_closure_reject_tol"],
          "Numerics.ale.remap_mass_closure_reject_tol");
      if (!(numerics.ale.remap_mass_closure_reject_tol >= 0.0)) {
        throw ConfigError(
            "Numerics.ale.remap_mass_closure_reject_tol must be >= 0");
      }
    }
    if (has_key(ale, "rezone_closure_cooldown_steps")) {
      numerics.ale.rezone_closure_cooldown_steps = strict_int32(
          ale["rezone_closure_cooldown_steps"],
          "Numerics.ale.rezone_closure_cooldown_steps");
      if (numerics.ale.rezone_closure_cooldown_steps < 1) {
        throw ConfigError(
            "Numerics.ale.rezone_closure_cooldown_steps must be >= 1");
      }
    }
    if (has_key(ale, "csr_optionb_coherent_enabled")) {
      numerics.ale.csr_optionb_coherent_enabled = strict_bool(
          ale["csr_optionb_coherent_enabled"],
          "Numerics.ale.csr_optionb_coherent_enabled");
    }
    if (has_key(ale, "csr_optionb_velocity_remap_enabled")) {
      numerics.ale.csr_optionb_velocity_remap_enabled = strict_bool(
          ale["csr_optionb_velocity_remap_enabled"],
          "Numerics.ale.csr_optionb_velocity_remap_enabled");
    }
    if (has_key(ale, "pole_axis_bbsw_enabled")) {
      numerics.ale.pole_axis_bbsw_enabled = strict_bool(
          ale["pole_axis_bbsw_enabled"],
          "Numerics.ale.pole_axis_bbsw_enabled");
    }
    if (has_key(ale, "axis_contact_guard_enabled")) {
      numerics.ale.axis_contact_guard_enabled = strict_bool(
          ale["axis_contact_guard_enabled"],
          "Numerics.ale.axis_contact_guard_enabled");
    }
    if (has_key(ale, "mass_floor_absorb_enabled")) {
      numerics.ale.mass_floor_absorb_enabled = strict_bool(
          ale["mass_floor_absorb_enabled"],
          "Numerics.ale.mass_floor_absorb_enabled");
    }
    if (has_key(ale, "interior_patch_remap_enabled")) {
      numerics.ale.interior_patch_remap_enabled = strict_bool(
          ale["interior_patch_remap_enabled"],
          "Numerics.ale.interior_patch_remap_enabled");
    }
    if (has_key(ale, "central_pseudo_core_ring_absorption_max_rings")) {
      numerics.ale.central_pseudo_core_ring_absorption_max_rings =
          strict_int32(ale["central_pseudo_core_ring_absorption_max_rings"],
                       "Numerics.ale.central_pseudo_core_ring_absorption_max_"
                       "rings");
      if (numerics.ale.central_pseudo_core_ring_absorption_max_rings < 0) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_ring_absorption_max_rings must "
            "be >= 0 (0 = unlimited up to the guard)");
      }
    }
    if (has_key(ale, "central_pseudo_core_ring_absorption_gas_tracer_min")) {
      numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_min =
          numeric_as_double(
              ale["central_pseudo_core_ring_absorption_gas_tracer_min"],
              "Numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_"
              "min");
      if (!(numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_min >
                0.0 &&
            numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_min <
                1.0)) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_min "
            "must be in (0, 1)");
      }
    }
    if (has_key(ale,
                "central_pseudo_core_ring_absorption_gas_tracer_cell_min")) {
      numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_cell_min =
          numeric_as_double(
              ale["central_pseudo_core_ring_absorption_gas_tracer_cell_min"],
              "Numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_"
              "cell_min");
      if (!(numerics.ale
                    .central_pseudo_core_ring_absorption_gas_tracer_cell_min >
                0.0 &&
            numerics.ale
                    .central_pseudo_core_ring_absorption_gas_tracer_cell_min <
                1.0)) {
        throw ConfigError(
            "Numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_"
            "cell_min must be in (0, 1)");
      }
    }
    if (has_key(ale, "pole_sector_rezone_enabled")) {
      numerics.ale.pole_sector_rezone_enabled = strict_bool(
          ale["pole_sector_rezone_enabled"],
          "Numerics.ale.pole_sector_rezone_enabled");
    }
    if (has_key(ale, "pole_sector_rezone_m_theta")) {
      numerics.ale.pole_sector_rezone_m_theta = strict_int32(
          ale["pole_sector_rezone_m_theta"],
          "Numerics.ale.pole_sector_rezone_m_theta");
      if (numerics.ale.pole_sector_rezone_m_theta < 2) {
        throw ConfigError(
            "Numerics.ale.pole_sector_rezone_m_theta must be >= 2");
      }
    }
    if (has_key(ale, "pole_sector_rezone_lambda")) {
      numerics.ale.pole_sector_rezone_lambda = numeric_as_double(
          ale["pole_sector_rezone_lambda"],
          "Numerics.ale.pole_sector_rezone_lambda");
      if (!(numerics.ale.pole_sector_rezone_lambda > 0.0 &&
            numerics.ale.pole_sector_rezone_lambda <= 1.0)) {
        throw ConfigError(
            "Numerics.ale.pole_sector_rezone_lambda must be in (0, 1]");
      }
    }
    if (has_key(ale, "pole_sector_rezone_mode")) {
      numerics.ale.pole_sector_rezone_mode = strict_string(
          ale["pole_sector_rezone_mode"],
          "Numerics.ale.pole_sector_rezone_mode");
      if (numerics.ale.pole_sector_rezone_mode != "uniform" &&
          numerics.ale.pole_sector_rezone_mode != "equal_mu") {
        throw ConfigError(
            "Numerics.ale.pole_sector_rezone_mode must be \"uniform\" or "
            "\"equal_mu\"");
      }
    }
    if (has_key(ale, "pole_sector_rezone_deadband_frac")) {
      numerics.ale.pole_sector_rezone_deadband_frac = numeric_as_double(
          ale["pole_sector_rezone_deadband_frac"],
          "Numerics.ale.pole_sector_rezone_deadband_frac");
      if (!(numerics.ale.pole_sector_rezone_deadband_frac >= 0.0 &&
            numerics.ale.pole_sector_rezone_deadband_frac < 1.0)) {
        throw ConfigError(
            "Numerics.ale.pole_sector_rezone_deadband_frac must be in "
            "[0, 1)");
      }
    }
    tenryu::core::validate_transaction_failure_inject_config(config);
  }

  if (has_key(kwargs, "plic")) {
    const py::handle plic_obj = kwargs["plic"];
    if (!py::isinstance<py::dict>(plic_obj)) {
      throw_value_type_error("Numerics.plic", "dict", plic_obj);
    }
    const py::dict plic = py::reinterpret_borrow<py::dict>(plic_obj);
    enforce_known_keys(plic, "Numerics.plic",
                       {"enabled",
                        "normal_estimator",
                        "t0_volume_cut_method",
                        "t0_volume_cut_max_depth",
                        "t0_volume_cut_volfrac_tol",
                        "fast_path_threshold_min",
                        "fast_path_threshold_max",
                        "fast_path_halo_radius_cells",
                        "alpha_solver_max_iter",
                        "alpha_tolerance_rel",
                        "thermodynamic_error_soft_threshold",
                        "thermodynamic_error_hard_threshold",
                        "class_d_dense_fraction_threshold",
                        "material_interface_per_cell_state",
                        "production_comparable_gate_strict",
                        "drift_sensor_max_relative",
                        "drift_sensor_max_swept_fraction",
                        "prev_normal_freshness_volfrac_threshold",
                        "plic_per_step_cost_target_fraction",
                        "in_run_disabled",
                        "rho_material_aware_donor"});
    auto& plic_cfg = numerics.plic;
    if (has_key(plic, "enabled")) {
      plic_cfg.enabled = strict_bool(plic["enabled"], "Numerics.plic.enabled");
    }
    if (has_key(plic, "normal_estimator")) {
      plic_cfg.normal_estimator =
          strict_string(plic["normal_estimator"], "Numerics.plic.normal_estimator");
      if (!is_plic_normal_estimator(plic_cfg.normal_estimator)) {
        throw ConfigError(
            "Numerics.plic.normal_estimator must be one of "
            "{\"youngs\", \"LVIRA\", \"youngs_seeded_LVIRA\"}");
      }
    }
    if (has_key(plic, "t0_volume_cut_method")) {
      plic_cfg.t0_volume_cut_method = strict_string(
          plic["t0_volume_cut_method"], "Numerics.plic.t0_volume_cut_method");
      if (!is_plic_t0_volume_cut_method(plic_cfg.t0_volume_cut_method)) {
        throw ConfigError(
            "Numerics.plic.t0_volume_cut_method must be one of "
            "{\"centroid_only_legacy\", \"adaptive_subdivision_2x2\", "
            "\"adaptive_subdivision_3x3\"}");
      }
    }
    if (has_key(plic, "t0_volume_cut_max_depth")) {
      plic_cfg.t0_volume_cut_max_depth = strict_int32(
          plic["t0_volume_cut_max_depth"], "Numerics.plic.t0_volume_cut_max_depth");
    }
    if (has_key(plic, "t0_volume_cut_volfrac_tol")) {
      plic_cfg.t0_volume_cut_volfrac_tol = numeric_as_double(
          plic["t0_volume_cut_volfrac_tol"], "Numerics.plic.t0_volume_cut_volfrac_tol");
    }
    if (has_key(plic, "fast_path_threshold_min")) {
      plic_cfg.fast_path_threshold_min = numeric_as_double(
          plic["fast_path_threshold_min"], "Numerics.plic.fast_path_threshold_min");
    }
    if (has_key(plic, "fast_path_threshold_max")) {
      plic_cfg.fast_path_threshold_max = numeric_as_double(
          plic["fast_path_threshold_max"], "Numerics.plic.fast_path_threshold_max");
    }
    if (has_key(plic, "fast_path_halo_radius_cells")) {
      plic_cfg.fast_path_halo_radius_cells = strict_int32(
          plic["fast_path_halo_radius_cells"],
          "Numerics.plic.fast_path_halo_radius_cells");
    }
    if (has_key(plic, "alpha_solver_max_iter")) {
      plic_cfg.alpha_solver_max_iter = strict_int32(
          plic["alpha_solver_max_iter"], "Numerics.plic.alpha_solver_max_iter");
    }
    if (has_key(plic, "alpha_tolerance_rel")) {
      plic_cfg.alpha_tolerance_rel = numeric_as_double(
          plic["alpha_tolerance_rel"], "Numerics.plic.alpha_tolerance_rel");
    }
    if (has_key(plic, "thermodynamic_error_soft_threshold")) {
      plic_cfg.thermodynamic_error_soft_threshold = numeric_as_double(
          plic["thermodynamic_error_soft_threshold"],
          "Numerics.plic.thermodynamic_error_soft_threshold");
    }
    if (has_key(plic, "thermodynamic_error_hard_threshold")) {
      plic_cfg.thermodynamic_error_hard_threshold = numeric_as_double(
          plic["thermodynamic_error_hard_threshold"],
          "Numerics.plic.thermodynamic_error_hard_threshold");
    }
    if (has_key(plic, "class_d_dense_fraction_threshold")) {
      plic_cfg.class_d_dense_fraction_threshold = numeric_as_double(
          plic["class_d_dense_fraction_threshold"],
          "Numerics.plic.class_d_dense_fraction_threshold");
    }
    if (has_key(plic, "material_interface_per_cell_state")) {
      plic_cfg.material_interface_per_cell_state = strict_string(
          plic["material_interface_per_cell_state"],
          "Numerics.plic.material_interface_per_cell_state");
      if (!is_plic_per_cell_state(plic_cfg.material_interface_per_cell_state)) {
        throw ConfigError(
            "Numerics.plic.material_interface_per_cell_state must be one of "
            "{\"off\", \"sparse_on_degradation\", \"dense_debug\"}");
      }
    }
    if (has_key(plic, "production_comparable_gate_strict")) {
      plic_cfg.production_comparable_gate_strict = strict_bool(
          plic["production_comparable_gate_strict"],
          "Numerics.plic.production_comparable_gate_strict");
    }
    if (has_key(plic, "drift_sensor_max_relative")) {
      plic_cfg.drift_sensor_max_relative = numeric_as_double(
          plic["drift_sensor_max_relative"], "Numerics.plic.drift_sensor_max_relative");
    }
    if (has_key(plic, "drift_sensor_max_swept_fraction")) {
      plic_cfg.drift_sensor_max_swept_fraction = numeric_as_double(
          plic["drift_sensor_max_swept_fraction"],
          "Numerics.plic.drift_sensor_max_swept_fraction");
    }
    if (has_key(plic, "prev_normal_freshness_volfrac_threshold")) {
      plic_cfg.prev_normal_freshness_volfrac_threshold = numeric_as_double(
          plic["prev_normal_freshness_volfrac_threshold"],
          "Numerics.plic.prev_normal_freshness_volfrac_threshold");
    }
    if (has_key(plic, "plic_per_step_cost_target_fraction")) {
      plic_cfg.plic_per_step_cost_target_fraction = numeric_as_double(
          plic["plic_per_step_cost_target_fraction"],
          "Numerics.plic.plic_per_step_cost_target_fraction");
    }
    if (has_key(plic, "in_run_disabled")) {
      plic_cfg.in_run_disabled =
          strict_bool(plic["in_run_disabled"], "Numerics.plic.in_run_disabled");
    }
    if (has_key(plic, "rho_material_aware_donor")) {
      plic_cfg.rho_material_aware_donor = strict_bool(
          plic["rho_material_aware_donor"],
          "Numerics.plic.rho_material_aware_donor");
    }
    if (plic_cfg.t0_volume_cut_max_depth < 4 ||
        plic_cfg.t0_volume_cut_max_depth > 16) {
      throw ValueError("Numerics.plic.t0_volume_cut_max_depth must be in [4, 16]");
    }
    ensure_positive_finite(plic_cfg.t0_volume_cut_volfrac_tol,
                           "Numerics.plic.t0_volume_cut_volfrac_tol");
    ensure_positive_finite(plic_cfg.fast_path_threshold_min,
                           "Numerics.plic.fast_path_threshold_min");
    ensure_positive_finite(plic_cfg.fast_path_threshold_max,
                           "Numerics.plic.fast_path_threshold_max");
    ensure_positive_finite(static_cast<double>(plic_cfg.fast_path_halo_radius_cells),
                           "Numerics.plic.fast_path_halo_radius_cells");
    ensure_positive_finite(static_cast<double>(plic_cfg.alpha_solver_max_iter),
                           "Numerics.plic.alpha_solver_max_iter");
    ensure_positive_finite(plic_cfg.alpha_tolerance_rel,
                           "Numerics.plic.alpha_tolerance_rel");
    ensure_positive_finite(plic_cfg.thermodynamic_error_soft_threshold,
                           "Numerics.plic.thermodynamic_error_soft_threshold");
    ensure_positive_finite(plic_cfg.thermodynamic_error_hard_threshold,
                           "Numerics.plic.thermodynamic_error_hard_threshold");
    ensure_positive_finite(plic_cfg.class_d_dense_fraction_threshold,
                           "Numerics.plic.class_d_dense_fraction_threshold");
    ensure_positive_finite(plic_cfg.drift_sensor_max_relative,
                           "Numerics.plic.drift_sensor_max_relative");
    ensure_positive_finite(plic_cfg.drift_sensor_max_swept_fraction,
                           "Numerics.plic.drift_sensor_max_swept_fraction");
    ensure_positive_finite(plic_cfg.prev_normal_freshness_volfrac_threshold,
                           "Numerics.plic.prev_normal_freshness_volfrac_threshold");
    ensure_positive_finite(plic_cfg.plic_per_step_cost_target_fraction,
                           "Numerics.plic.plic_per_step_cost_target_fraction");
  }

  if (has_key(kwargs, "materials")) {
    const py::handle materials_obj = kwargs["materials"];
    if (!py::isinstance<py::dict>(materials_obj)) {
      throw_value_type_error("Numerics.materials", "dict", materials_obj);
    }
    const py::dict materials = py::reinterpret_borrow<py::dict>(materials_obj);
    enforce_known_keys(materials, "Numerics.materials",
                       {"per_material_conservation_enabled",
                        "presence_threshold_volfrac",
                        "presence_threshold_mass_density_g_per_cc",
                        "eos_table_validity_lower_bound_g_per_cc",
                        "lazy_cache_te_m_enabled",
                        "hdf5_emit_derived_per_material",
                        "deposit_redistribute_fallback_enabled",
                        "deposit_redistribute_provenance_label",
                        "conservation_residual_warn_threshold_rel",
                        "conservation_residual_hard_warning_threshold_rel"});
    auto& materials_cfg = numerics.materials;
    if (has_key(materials, "per_material_conservation_enabled")) {
      materials_cfg.per_material_conservation_enabled = strict_bool(
          materials["per_material_conservation_enabled"],
          "Numerics.materials.per_material_conservation_enabled");
    }
    if (has_key(materials, "presence_threshold_volfrac")) {
      materials_cfg.presence_threshold_volfrac = numeric_as_double(
          materials["presence_threshold_volfrac"],
          "Numerics.materials.presence_threshold_volfrac");
    }
    if (has_key(materials, "presence_threshold_mass_density_g_per_cc")) {
      materials_cfg.presence_threshold_mass_density_g_per_cc = numeric_as_double(
          materials["presence_threshold_mass_density_g_per_cc"],
          "Numerics.materials.presence_threshold_mass_density_g_per_cc");
    }
    if (has_key(materials, "eos_table_validity_lower_bound_g_per_cc")) {
      const py::handle map_obj = materials["eos_table_validity_lower_bound_g_per_cc"];
      if (!py::isinstance<py::dict>(map_obj)) {
        throw_value_type_error(
            "Numerics.materials.eos_table_validity_lower_bound_g_per_cc",
            "dict",
            map_obj);
      }
      materials_cfg.eos_table_validity_lower_bound_g_per_cc.clear();
      const py::dict bounds = py::reinterpret_borrow<py::dict>(map_obj);
      for (const auto item : bounds) {
        const std::string name = py::str(item.first).cast<std::string>();
        const double lower = numeric_as_double(
            item.second,
            "Numerics.materials.eos_table_validity_lower_bound_g_per_cc." + name);
        ensure_positive_finite(
            lower,
            "Numerics.materials.eos_table_validity_lower_bound_g_per_cc." + name);
        materials_cfg.eos_table_validity_lower_bound_g_per_cc[name] = lower;
      }
    }
    if (has_key(materials, "lazy_cache_te_m_enabled")) {
      materials_cfg.lazy_cache_te_m_enabled = strict_bool(
          materials["lazy_cache_te_m_enabled"],
          "Numerics.materials.lazy_cache_te_m_enabled");
    }
    if (has_key(materials, "hdf5_emit_derived_per_material")) {
      materials_cfg.hdf5_emit_derived_per_material = strict_bool(
          materials["hdf5_emit_derived_per_material"],
          "Numerics.materials.hdf5_emit_derived_per_material");
    }
    if (has_key(materials, "deposit_redistribute_fallback_enabled")) {
      materials_cfg.deposit_redistribute_fallback_enabled = strict_bool(
          materials["deposit_redistribute_fallback_enabled"],
          "Numerics.materials.deposit_redistribute_fallback_enabled");
    }
    if (has_key(materials, "deposit_redistribute_provenance_label")) {
      materials_cfg.deposit_redistribute_provenance_label = strict_string(
          materials["deposit_redistribute_provenance_label"],
          "Numerics.materials.deposit_redistribute_provenance_label");
    }
    if (has_key(materials, "conservation_residual_warn_threshold_rel")) {
      materials_cfg.conservation_residual_warn_threshold_rel = numeric_as_double(
          materials["conservation_residual_warn_threshold_rel"],
          "Numerics.materials.conservation_residual_warn_threshold_rel");
    }
    if (has_key(materials, "conservation_residual_hard_warning_threshold_rel")) {
      materials_cfg.conservation_residual_hard_warning_threshold_rel = numeric_as_double(
          materials["conservation_residual_hard_warning_threshold_rel"],
          "Numerics.materials.conservation_residual_hard_warning_threshold_rel");
    }
    ensure_positive_finite(materials_cfg.presence_threshold_volfrac,
                           "Numerics.materials.presence_threshold_volfrac");
    ensure_positive_finite(
        materials_cfg.presence_threshold_mass_density_g_per_cc,
        "Numerics.materials.presence_threshold_mass_density_g_per_cc");
    ensure_positive_finite(
        materials_cfg.conservation_residual_warn_threshold_rel,
        "Numerics.materials.conservation_residual_warn_threshold_rel");
    ensure_positive_finite(
        materials_cfg.conservation_residual_hard_warning_threshold_rel,
        "Numerics.materials.conservation_residual_hard_warning_threshold_rel");
    if (materials_cfg.conservation_residual_hard_warning_threshold_rel <
        materials_cfg.conservation_residual_warn_threshold_rel) {
      throw ValueError(
          "Numerics.materials.conservation_residual_hard_warning_threshold_rel "
          "must be >= conservation_residual_warn_threshold_rel");
    }
    if (materials_cfg.deposit_redistribute_provenance_label.empty()) {
      throw ValueError(
          "Numerics.materials.deposit_redistribute_provenance_label must be non-empty");
    }
  }

  if (has_key(kwargs, "ale1d")) {
    const py::handle ale1d_obj = kwargs["ale1d"];
    if (!py::isinstance<py::dict>(ale1d_obj)) {
      throw_value_type_error("Numerics.ale1d", "dict", ale1d_obj);
    }
    const py::dict ale1d = py::reinterpret_borrow<py::dict>(ale1d_obj);
    enforce_known_keys(ale1d, "Numerics.ale1d",
                       {"enabled", "every_n_steps", "min_steps_between_ale",
                        "enable_benefit_gate", "benefit_min_dt_gain",
                        "candidate_dt_penalty_max", "emergency_enabled",
                        "min_cells", "protected_fraction_max",
                        "min_movable_segment_warn", "min_movable_segment_hard",
                        "max_node_displacement_fraction_mu",
                        "max_node_displacement_fraction_r",
                        "ke_conservation_closure",
                        "total_mass_tol", "material_mass_tol",
                        "radiation_group_energy_tol",
                        "material_internal_energy_tol",
                        "total_material_energy_tol", "global_total_energy_tol",
                        "kinetic_energy_drift_tol",
                        "diagnostics_enabled",
                        "diagnostics_log_every_n_steps",
                        "diagnostics_collect_step_result",
                        "diagnostics_fail_on_unexpected_apply",
                        "laser_sensor", "ablation_sensor", "shock_sensor",
                        "interface_sensor", "center_sensor", "rezone",
                        "min_width_floor", "remap"});
    auto& ale1d_cfg = numerics.ale1d;
    const auto parse_tol = [&](const char* key, auto& tol) {
      if (!has_key(ale1d, key)) {
        return;
      }
      const py::handle tol_obj = ale1d[key];
      const std::string path = std::string("Numerics.ale1d.") + key;
      if (!py::isinstance<py::dict>(tol_obj)) {
        throw_value_type_error(path, "dict", tol_obj);
      }
      const py::dict tol_dict = py::reinterpret_borrow<py::dict>(tol_obj);
      enforce_known_keys(tol_dict, path, {"soft", "hard"});
      if (has_key(tol_dict, "soft")) {
        tol.soft = numeric_as_double(tol_dict["soft"], path + ".soft");
      }
      if (has_key(tol_dict, "hard")) {
        tol.hard = numeric_as_double(tol_dict["hard"], path + ".hard");
      }
    };
    const auto parse_sensor_dict = [&](const char* key, const auto& parse_contents) {
      if (!has_key(ale1d, key)) {
        return;
      }
      const py::handle sensor_obj = ale1d[key];
      const std::string path = std::string("Numerics.ale1d.") + key;
      if (!py::isinstance<py::dict>(sensor_obj)) {
        throw_value_type_error(path, "dict", sensor_obj);
      }
      const py::dict sensor = py::reinterpret_borrow<py::dict>(sensor_obj);
      parse_contents(sensor, path);
    };
    const auto parse_rezone_dict = [&]() {
      if (!has_key(ale1d, "rezone")) {
        return;
      }
      const py::handle rezone_obj = ale1d["rezone"];
      const std::string path = "Numerics.ale1d.rezone";
      if (!py::isinstance<py::dict>(rezone_obj)) {
        throw_value_type_error(path, "dict", rezone_obj);
      }
      const py::dict rezone = py::reinterpret_borrow<py::dict>(rezone_obj);
      enforce_known_keys(
          rezone, path,
          {"monitor_floor", "monitor_wmax_ratio",
           "monitor_smoothing_iterations",
           "monitor_smooth_across_protected_faces", "min_floor_fraction",
           "gaussian_truncation_sigma", "spatial_monitor_enabled",
           "spatial_target_cells_fraction", "spatial_power",
           "laser_spatial_dr_min_cm", "laser_spatial_dr_max_cm",
           "ablation_spatial_dr_min_cm", "ablation_spatial_dr_max_cm",
           "shock_spatial_dr_min_cm", "shock_spatial_dr_max_cm"});
      auto& cfg = ale1d_cfg.rezone;
      if (has_key(rezone, "monitor_floor")) {
        cfg.monitor_floor =
            numeric_as_double(rezone["monitor_floor"], path + ".monitor_floor");
      }
      if (has_key(rezone, "monitor_wmax_ratio")) {
        cfg.monitor_wmax_ratio =
            numeric_as_double(rezone["monitor_wmax_ratio"],
                              path + ".monitor_wmax_ratio");
      }
      if (has_key(rezone, "monitor_smoothing_iterations")) {
        cfg.monitor_smoothing_iterations = strict_int32(
            rezone["monitor_smoothing_iterations"],
            path + ".monitor_smoothing_iterations");
      }
      if (has_key(rezone, "monitor_smooth_across_protected_faces")) {
        cfg.monitor_smooth_across_protected_faces = strict_bool(
            rezone["monitor_smooth_across_protected_faces"],
            path + ".monitor_smooth_across_protected_faces");
      }
      if (has_key(rezone, "min_floor_fraction")) {
        cfg.min_floor_fraction =
            numeric_as_double(rezone["min_floor_fraction"],
                              path + ".min_floor_fraction");
      }
      if (has_key(rezone, "gaussian_truncation_sigma")) {
        cfg.gaussian_truncation_sigma = numeric_as_double(
            rezone["gaussian_truncation_sigma"],
            path + ".gaussian_truncation_sigma");
      }
      if (has_key(rezone, "spatial_monitor_enabled")) {
        cfg.spatial_monitor_enabled = strict_bool(
            rezone["spatial_monitor_enabled"],
            path + ".spatial_monitor_enabled");
      }
      if (has_key(rezone, "spatial_target_cells_fraction")) {
        cfg.spatial_target_cells_fraction = numeric_as_double(
            rezone["spatial_target_cells_fraction"],
            path + ".spatial_target_cells_fraction");
      }
      if (has_key(rezone, "spatial_power")) {
        cfg.spatial_power =
            numeric_as_double(rezone["spatial_power"], path + ".spatial_power");
      }
      if (has_key(rezone, "laser_spatial_dr_min_cm")) {
        cfg.laser_spatial_dr_min_cm = numeric_as_double(
            rezone["laser_spatial_dr_min_cm"],
            path + ".laser_spatial_dr_min_cm");
      }
      if (has_key(rezone, "laser_spatial_dr_max_cm")) {
        cfg.laser_spatial_dr_max_cm = numeric_as_double(
            rezone["laser_spatial_dr_max_cm"],
            path + ".laser_spatial_dr_max_cm");
      }
      if (has_key(rezone, "ablation_spatial_dr_min_cm")) {
        cfg.ablation_spatial_dr_min_cm = numeric_as_double(
            rezone["ablation_spatial_dr_min_cm"],
            path + ".ablation_spatial_dr_min_cm");
      }
      if (has_key(rezone, "ablation_spatial_dr_max_cm")) {
        cfg.ablation_spatial_dr_max_cm = numeric_as_double(
            rezone["ablation_spatial_dr_max_cm"],
            path + ".ablation_spatial_dr_max_cm");
      }
      if (has_key(rezone, "shock_spatial_dr_min_cm")) {
        cfg.shock_spatial_dr_min_cm = numeric_as_double(
            rezone["shock_spatial_dr_min_cm"],
            path + ".shock_spatial_dr_min_cm");
      }
      if (has_key(rezone, "shock_spatial_dr_max_cm")) {
        cfg.shock_spatial_dr_max_cm = numeric_as_double(
            rezone["shock_spatial_dr_max_cm"],
            path + ".shock_spatial_dr_max_cm");
      }
    };
    const auto parse_min_width_floor_dict = [&]() {
      if (!has_key(ale1d, "min_width_floor")) {
        return;
      }
      const py::handle floor_obj = ale1d["min_width_floor"];
      const std::string path = "Numerics.ale1d.min_width_floor";
      if (!py::isinstance<py::dict>(floor_obj)) {
        throw_value_type_error(path, "dict", floor_obj);
      }
      const py::dict floor = py::reinterpret_borrow<py::dict>(floor_obj);
      enforce_known_keys(
          floor, path,
          {"enabled", "floor_cm", "target_factor", "relief_halfwidth_cells",
           "max_growth_factor", "retrigger_cooldown_steps"});
      auto& cfg = ale1d_cfg.min_width_floor;
      if (has_key(floor, "enabled")) {
        cfg.enabled = strict_bool(floor["enabled"], path + ".enabled");
      }
      if (has_key(floor, "floor_cm")) {
        cfg.floor_cm = numeric_as_double(floor["floor_cm"], path + ".floor_cm");
      }
      if (has_key(floor, "target_factor")) {
        cfg.target_factor = numeric_as_double(
            floor["target_factor"], path + ".target_factor");
      }
      if (has_key(floor, "relief_halfwidth_cells")) {
        cfg.relief_halfwidth_cells = strict_int32(
            floor["relief_halfwidth_cells"],
            path + ".relief_halfwidth_cells");
      }
      if (has_key(floor, "max_growth_factor")) {
        cfg.max_growth_factor = numeric_as_double(
            floor["max_growth_factor"], path + ".max_growth_factor");
      }
      if (has_key(floor, "retrigger_cooldown_steps")) {
        cfg.retrigger_cooldown_steps = strict_int32(
            floor["retrigger_cooldown_steps"],
            path + ".retrigger_cooldown_steps");
      }
    };
    const auto parse_remap_dict = [&]() {
      if (!has_key(ale1d, "remap")) {
        return;
      }
      const py::handle remap_obj = ale1d["remap"];
      const std::string path = "Numerics.ale1d.remap";
      if (!py::isinstance<py::dict>(remap_obj)) {
        throw_value_type_error(path, "dict", remap_obj);
      }
      const py::dict remap = py::reinterpret_borrow<py::dict>(remap_obj);
      enforce_known_keys(
          remap, path,
          {"reject_multicell_sweeps", "high_order_enabled",
           "limiter_theta", "high_order_ramp_cells",
           "radiation_high_order_ramp_cells",
           "fallback_to_first_order_on_bounds_fail",
           "reject_strict_zero_flux_on_moving_protected_face"});
      auto& cfg = ale1d_cfg.remap;
      if (has_key(remap, "reject_multicell_sweeps")) {
        cfg.reject_multicell_sweeps = strict_bool(
            remap["reject_multicell_sweeps"],
            path + ".reject_multicell_sweeps");
      }
      if (has_key(remap, "high_order_enabled")) {
        cfg.high_order_enabled =
            strict_bool(remap["high_order_enabled"],
                        path + ".high_order_enabled");
      }
      if (has_key(remap, "limiter_theta")) {
        cfg.limiter_theta =
            numeric_as_double(remap["limiter_theta"], path + ".limiter_theta");
      }
      if (has_key(remap, "high_order_ramp_cells")) {
        cfg.high_order_ramp_cells = strict_int32(
            remap["high_order_ramp_cells"], path + ".high_order_ramp_cells");
      }
      if (has_key(remap, "radiation_high_order_ramp_cells")) {
        cfg.radiation_high_order_ramp_cells =
            strict_int32(remap["radiation_high_order_ramp_cells"],
                         path + ".radiation_high_order_ramp_cells");
      }
      if (has_key(remap, "fallback_to_first_order_on_bounds_fail")) {
        cfg.fallback_to_first_order_on_bounds_fail = strict_bool(
            remap["fallback_to_first_order_on_bounds_fail"],
            path + ".fallback_to_first_order_on_bounds_fail");
      }
      if (has_key(remap, "reject_strict_zero_flux_on_moving_protected_face")) {
        cfg.reject_strict_zero_flux_on_moving_protected_face = strict_bool(
            remap["reject_strict_zero_flux_on_moving_protected_face"],
            path + ".reject_strict_zero_flux_on_moving_protected_face");
      }
    };

    if (has_key(ale1d, "enabled")) {
      ale1d_cfg.enabled = strict_bool(ale1d["enabled"], "Numerics.ale1d.enabled");
    }
    if (has_key(ale1d, "every_n_steps")) {
      ale1d_cfg.every_n_steps =
          strict_int32(ale1d["every_n_steps"], "Numerics.ale1d.every_n_steps");
    }
    if (has_key(ale1d, "min_steps_between_ale")) {
      ale1d_cfg.min_steps_between_ale = strict_int32(
          ale1d["min_steps_between_ale"], "Numerics.ale1d.min_steps_between_ale");
    }
    if (has_key(ale1d, "enable_benefit_gate")) {
      ale1d_cfg.enable_benefit_gate = strict_bool(
          ale1d["enable_benefit_gate"], "Numerics.ale1d.enable_benefit_gate");
    }
    if (has_key(ale1d, "benefit_min_dt_gain")) {
      ale1d_cfg.benefit_min_dt_gain = numeric_as_double(
          ale1d["benefit_min_dt_gain"], "Numerics.ale1d.benefit_min_dt_gain");
    }
    if (has_key(ale1d, "candidate_dt_penalty_max")) {
      ale1d_cfg.candidate_dt_penalty_max = numeric_as_double(
          ale1d["candidate_dt_penalty_max"], "Numerics.ale1d.candidate_dt_penalty_max");
    }
    if (has_key(ale1d, "emergency_enabled")) {
      ale1d_cfg.emergency_enabled = strict_bool(
          ale1d["emergency_enabled"], "Numerics.ale1d.emergency_enabled");
    }
    if (has_key(ale1d, "min_cells")) {
      ale1d_cfg.min_cells =
          strict_int32(ale1d["min_cells"], "Numerics.ale1d.min_cells");
    }
    if (has_key(ale1d, "protected_fraction_max")) {
      ale1d_cfg.protected_fraction_max = numeric_as_double(
          ale1d["protected_fraction_max"], "Numerics.ale1d.protected_fraction_max");
    }
    if (has_key(ale1d, "min_movable_segment_warn")) {
      ale1d_cfg.min_movable_segment_warn = strict_int32(
          ale1d["min_movable_segment_warn"], "Numerics.ale1d.min_movable_segment_warn");
    }
    if (has_key(ale1d, "min_movable_segment_hard")) {
      ale1d_cfg.min_movable_segment_hard = strict_int32(
          ale1d["min_movable_segment_hard"], "Numerics.ale1d.min_movable_segment_hard");
    }
    if (has_key(ale1d, "max_node_displacement_fraction_mu")) {
      ale1d_cfg.max_node_displacement_fraction_mu = numeric_as_double(
          ale1d["max_node_displacement_fraction_mu"],
          "Numerics.ale1d.max_node_displacement_fraction_mu");
    }
    if (has_key(ale1d, "max_node_displacement_fraction_r")) {
      ale1d_cfg.max_node_displacement_fraction_r = numeric_as_double(
          ale1d["max_node_displacement_fraction_r"],
          "Numerics.ale1d.max_node_displacement_fraction_r");
    }
    if (has_key(ale1d, "ke_conservation_closure")) {
      ale1d_cfg.ke_conservation_closure = strict_bool(
          ale1d["ke_conservation_closure"],
          "Numerics.ale1d.ke_conservation_closure");
    }
    parse_tol("total_mass_tol", ale1d_cfg.total_mass_tol);
    parse_tol("material_mass_tol", ale1d_cfg.material_mass_tol);
    parse_tol("radiation_group_energy_tol", ale1d_cfg.radiation_group_energy_tol);
    parse_tol("material_internal_energy_tol", ale1d_cfg.material_internal_energy_tol);
    parse_tol("total_material_energy_tol", ale1d_cfg.total_material_energy_tol);
    parse_tol("global_total_energy_tol", ale1d_cfg.global_total_energy_tol);
    parse_tol("kinetic_energy_drift_tol", ale1d_cfg.kinetic_energy_drift_tol);
    if (has_key(ale1d, "diagnostics_enabled")) {
      ale1d_cfg.diagnostics_enabled = strict_bool(
          ale1d["diagnostics_enabled"], "Numerics.ale1d.diagnostics_enabled");
    }
    if (has_key(ale1d, "diagnostics_log_every_n_steps")) {
      ale1d_cfg.diagnostics_log_every_n_steps = strict_int32(
          ale1d["diagnostics_log_every_n_steps"],
          "Numerics.ale1d.diagnostics_log_every_n_steps");
    }
    if (has_key(ale1d, "diagnostics_collect_step_result")) {
      ale1d_cfg.diagnostics_collect_step_result = strict_bool(
          ale1d["diagnostics_collect_step_result"],
          "Numerics.ale1d.diagnostics_collect_step_result");
    }
    if (has_key(ale1d, "diagnostics_fail_on_unexpected_apply")) {
      ale1d_cfg.diagnostics_fail_on_unexpected_apply = strict_bool(
          ale1d["diagnostics_fail_on_unexpected_apply"],
          "Numerics.ale1d.diagnostics_fail_on_unexpected_apply");
    }
    parse_sensor_dict("laser_sensor", [&](const py::dict& sensor,
                                           const std::string& path) {
      auto& cfg = ale1d_cfg.laser_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "sigma_min_cells",
                          "sigma_max_cells", "peak_fraction", "conf_low", "conf_high"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "peak_fraction")) {
        cfg.peak_fraction = numeric_as_double(sensor["peak_fraction"], path + ".peak_fraction");
      }
      if (has_key(sensor, "conf_low")) {
        cfg.conf_low = numeric_as_double(sensor["conf_low"], path + ".conf_low");
      }
      if (has_key(sensor, "conf_high")) {
        cfg.conf_high = numeric_as_double(sensor["conf_high"], path + ".conf_high");
      }
    });
    parse_sensor_dict("ablation_sensor", [&](const py::dict& sensor,
                                              const std::string& path) {
      auto& cfg = ale1d_cfg.ablation_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "sigma_min_cells",
                          "sigma_max_cells", "peak_fraction", "reference_density_gcc",
                          "rho_gate_frac", "rho_gate_width", "te_gate_low_eV",
                          "te_gate_high_eV", "conf_low", "conf_high"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "peak_fraction")) {
        cfg.peak_fraction = numeric_as_double(sensor["peak_fraction"], path + ".peak_fraction");
      }
      if (has_key(sensor, "reference_density_gcc")) {
        cfg.reference_density_gcc =
            numeric_as_double(sensor["reference_density_gcc"], path + ".reference_density_gcc");
      }
      if (has_key(sensor, "rho_gate_frac")) {
        cfg.rho_gate_frac = numeric_as_double(sensor["rho_gate_frac"], path + ".rho_gate_frac");
      }
      if (has_key(sensor, "rho_gate_width")) {
        cfg.rho_gate_width =
            numeric_as_double(sensor["rho_gate_width"], path + ".rho_gate_width");
      }
      if (has_key(sensor, "te_gate_low_eV")) {
        cfg.te_gate_low_eV =
            numeric_as_double(sensor["te_gate_low_eV"], path + ".te_gate_low_eV");
      }
      if (has_key(sensor, "te_gate_high_eV")) {
        cfg.te_gate_high_eV =
            numeric_as_double(sensor["te_gate_high_eV"], path + ".te_gate_high_eV");
      }
      if (has_key(sensor, "conf_low")) {
        cfg.conf_low = numeric_as_double(sensor["conf_low"], path + ".conf_low");
      }
      if (has_key(sensor, "conf_high")) {
        cfg.conf_high = numeric_as_double(sensor["conf_high"], path + ".conf_high");
      }
    });
    parse_sensor_dict("shock_sensor", [&](const py::dict& sensor,
                                           const std::string& path) {
      auto& cfg = ale1d_cfg.shock_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "sigma_min_cells",
                          "sigma_max_cells", "peak_fraction", "qvisc_conf_low",
                          "qvisc_conf_high", "du_cs_conf_low", "du_cs_conf_high"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "peak_fraction")) {
        cfg.peak_fraction = numeric_as_double(sensor["peak_fraction"], path + ".peak_fraction");
      }
      if (has_key(sensor, "qvisc_conf_low")) {
        cfg.qvisc_conf_low =
            numeric_as_double(sensor["qvisc_conf_low"], path + ".qvisc_conf_low");
      }
      if (has_key(sensor, "qvisc_conf_high")) {
        cfg.qvisc_conf_high =
            numeric_as_double(sensor["qvisc_conf_high"], path + ".qvisc_conf_high");
      }
      if (has_key(sensor, "du_cs_conf_low")) {
        cfg.du_cs_conf_low =
            numeric_as_double(sensor["du_cs_conf_low"], path + ".du_cs_conf_low");
      }
      if (has_key(sensor, "du_cs_conf_high")) {
        cfg.du_cs_conf_high =
            numeric_as_double(sensor["du_cs_conf_high"], path + ".du_cs_conf_high");
      }
    });
    parse_sensor_dict("interface_sensor", [&](const py::dict& sensor,
                                               const std::string& path) {
      auto& cfg = ale1d_cfg.interface_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "target_cells_cap_fraction",
                          "max_features", "min_separation_cells", "jump_low",
                          "jump_high", "sigma_min_cells", "sigma_max_cells",
                          "pin_interfaces"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "target_cells_cap_fraction")) {
        cfg.target_cells_cap_fraction = numeric_as_double(
            sensor["target_cells_cap_fraction"], path + ".target_cells_cap_fraction");
      }
      if (has_key(sensor, "max_features")) {
        cfg.max_features = strict_int32(sensor["max_features"], path + ".max_features");
      }
      if (has_key(sensor, "min_separation_cells")) {
        cfg.min_separation_cells =
            strict_int32(sensor["min_separation_cells"], path + ".min_separation_cells");
      }
      if (has_key(sensor, "jump_low")) {
        cfg.jump_low = numeric_as_double(sensor["jump_low"], path + ".jump_low");
      }
      if (has_key(sensor, "jump_high")) {
        cfg.jump_high = numeric_as_double(sensor["jump_high"], path + ".jump_high");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "pin_interfaces")) {
        cfg.pin_interfaces = strict_bool(sensor["pin_interfaces"], path + ".pin_interfaces");
      }
    });
    parse_sensor_dict("center_sensor", [&](const py::dict& sensor,
                                            const std::string& path) {
      auto& cfg = ale1d_cfg.center_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "sigma_min_cells",
                          "sigma_max_cells", "search_x"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "search_x")) {
        cfg.search_x = numeric_as_double(sensor["search_x"], path + ".search_x");
      }
    });
    parse_rezone_dict();
    parse_min_width_floor_dict();
    parse_remap_dict();
  }

  if (has_key(kwargs, "floors")) {
    const py::handle floors_obj = kwargs["floors"];
    if (!py::isinstance<py::dict>(floors_obj)) {
      throw_value_type_error("Numerics.floors", "dict", floors_obj);
    }
    const py::dict floors = py::reinterpret_borrow<py::dict>(floors_obj);
    enforce_known_keys(floors, "Numerics.floors",
                       {"rho_floor_gcc", "Te_floor_eV", "Ti_floor_eV"});
    if (has_key(floors, "rho_floor_gcc")) {
      numerics.floors.rho =
          numeric_as_double(floors["rho_floor_gcc"], "Numerics.floors.rho_floor_gcc");
    }
    if (has_key(floors, "Te_floor_eV")) {
      numerics.floors.Te =
          numeric_as_double(floors["Te_floor_eV"], "Numerics.floors.Te_floor_eV");
    }
    if (has_key(floors, "Ti_floor_eV")) {
      numerics.floors.Ti =
          numeric_as_double(floors["Ti_floor_eV"], "Numerics.floors.Ti_floor_eV");
    }
  }

  if (has_key(kwargs, "positivity")) {
    const py::handle positivity_obj = kwargs["positivity"];
    if (!py::isinstance<py::dict>(positivity_obj)) {
      throw_value_type_error("Numerics.positivity", "dict", positivity_obj);
    }
    const py::dict positivity = py::reinterpret_borrow<py::dict>(positivity_obj);
    enforce_known_keys(positivity, "Numerics.positivity",
                       {"clamp", "Te_min_eV", "Ti_min_eV", "rho_floor_gcc"});
    if (has_key(positivity, "clamp")) {
      numerics.positivity_clamp =
          strict_bool(positivity["clamp"], "Numerics.positivity.clamp");
    }
    if (has_key(positivity, "rho_floor_gcc")) {
      numerics.floors.rho = std::max(
          numerics.floors.rho,
          numeric_as_double(positivity["rho_floor_gcc"],
                            "Numerics.positivity.rho_floor_gcc"));
    }
    if (has_key(positivity, "Te_min_eV")) {
      const double legacy =
          numeric_as_double(positivity["Te_min_eV"], "Numerics.positivity.Te_min_eV");
      tenryu::core::log_warning(
          "Numerics.positivity.Te_min_eV is deprecated; mapped to floors.Te");
      numerics.floors.Te = std::max(numerics.floors.Te, legacy);
    }
    if (has_key(positivity, "Ti_min_eV")) {
      numerics.floors.Ti = std::max(
          numerics.floors.Ti,
          numeric_as_double(positivity["Ti_min_eV"],
                            "Numerics.positivity.Ti_min_eV"));
    }
  }

  if (has_key(kwargs, "safety")) {
    const py::handle safety_obj = kwargs["safety"];
    if (!py::isinstance<py::dict>(safety_obj)) {
      throw_value_type_error("Numerics.safety", "dict", safety_obj);
    }
    const py::dict safety = py::reinterpret_borrow<py::dict>(safety_obj);
    enforce_known_keys(safety, "Numerics.safety",
                       {"energy_fatal", "nan_fatal", "energy_threshold",
                        "energy_budget_tol", "opacity_floor", "opacity_cap",
                        "clamp_warn_threshold", "clamp_fatal_threshold", "overshoot_warn",
                        "overshoot_fatal", "overshoot_fatal_enabled", "cell_search_fatal"});
    if (has_key(safety, "energy_fatal")) {
      numerics.safety.energy_fatal =
          strict_bool(safety["energy_fatal"], "Numerics.safety.energy_fatal");
    }
    if (has_key(safety, "nan_fatal")) {
      numerics.safety.nan_fatal =
          strict_bool(safety["nan_fatal"], "Numerics.safety.nan_fatal");
    }
    if (has_key(safety, "energy_threshold")) {
      numerics.safety.energy_budget_tol = numeric_as_double(
          safety["energy_threshold"], "Numerics.safety.energy_threshold");
    }
    if (has_key(safety, "energy_budget_tol")) {
      numerics.safety.energy_budget_tol = numeric_as_double(
          safety["energy_budget_tol"], "Numerics.safety.energy_budget_tol");
    }
    if (has_key(safety, "opacity_floor")) {
      numerics.safety.opacity_floor =
          numeric_as_double(safety["opacity_floor"], "Numerics.safety.opacity_floor");
    }
    if (has_key(safety, "opacity_cap")) {
      numerics.safety.opacity_cap =
          numeric_as_double(safety["opacity_cap"], "Numerics.safety.opacity_cap");
    }
    if (has_key(safety, "clamp_warn_threshold")) {
      numerics.safety.clamp_warn_threshold = strict_int32(
          safety["clamp_warn_threshold"],
          "Numerics.safety.clamp_warn_threshold");
    }
    if (has_key(safety, "clamp_fatal_threshold")) {
      numerics.safety.clamp_fatal_threshold = strict_int32(
          safety["clamp_fatal_threshold"],
          "Numerics.safety.clamp_fatal_threshold");
    }
    if (has_key(safety, "overshoot_warn")) {
      numerics.safety.overshoot_warn =
          numeric_as_double(safety["overshoot_warn"], "Numerics.safety.overshoot_warn");
    }
    if (has_key(safety, "overshoot_fatal")) {
      numerics.safety.overshoot_fatal = numeric_as_double(
          safety["overshoot_fatal"], "Numerics.safety.overshoot_fatal");
    }
    if (has_key(safety, "overshoot_fatal_enabled")) {
      numerics.safety.overshoot_fatal_enabled =
          strict_bool(safety["overshoot_fatal_enabled"],
                      "Numerics.safety.overshoot_fatal_enabled");
    }
    if (has_key(safety, "cell_search_fatal")) {
      tenryu::core::log_warning(
          "Numerics.safety.cell_search_fatal is accepted for compatibility and ignored in M01");
    }
  }
  validate_production_audit_config(numerics);
}

void Builder::set_output(py::dict kwargs) {
  mark_block_called(Block::Output);
  enforce_known_keys(kwargs, "Output",
                     {"directory", "format", "plot_every", "history_every",
                      "checkpoint_every", "plot_every_s", "history_every_s",
                      "checkpoint_every_s", "write_final_snapshot",
                      "checkpoint_keep_last", "compression", "compression_level",
                      "save_namelist_copy", "save_frozen_config", "plot_fields"});

  auto& output = config.output;
  if (has_key(kwargs, "directory")) {
    output.directory = strict_string(kwargs["directory"], "Output.directory");
  }
  if (has_key(kwargs, "format")) {
    output.format = strict_string(kwargs["format"], "Output.format");
  }
  if (has_key(kwargs, "plot_every")) {
    output.plot_every = strict_int32(kwargs["plot_every"], "Output.plot_every");
    output.plot_every_explicit = true;
  }
  if (has_key(kwargs, "history_every")) {
    output.history_every =
        strict_int32(kwargs["history_every"], "Output.history_every");
    output.history_every_explicit = true;
  }
  if (has_key(kwargs, "checkpoint_every")) {
    output.checkpoint_every =
        strict_int32(kwargs["checkpoint_every"], "Output.checkpoint_every");
    output.checkpoint_every_explicit = true;
  }
  if (has_key(kwargs, "plot_every_s")) {
    output.plot_every_s =
        numeric_as_double(kwargs["plot_every_s"], "Output.plot_every_s");
  }
  if (has_key(kwargs, "write_final_snapshot")) {
    output.write_final_snapshot = strict_bool(
        kwargs["write_final_snapshot"], "Output.write_final_snapshot");
  }
  if (has_key(kwargs, "history_every_s")) {
    output.history_every_s =
        numeric_as_double(kwargs["history_every_s"], "Output.history_every_s");
  }
  if (has_key(kwargs, "checkpoint_every_s")) {
    output.checkpoint_every_s =
        numeric_as_double(kwargs["checkpoint_every_s"], "Output.checkpoint_every_s");
  }
  if (has_key(kwargs, "checkpoint_keep_last")) {
    output.checkpoint_keep_last =
        strict_int32(kwargs["checkpoint_keep_last"], "Output.checkpoint_keep_last");
  }
  if (has_key(kwargs, "compression")) {
    output.compression = strict_string(kwargs["compression"], "Output.compression");
  }
  if (has_key(kwargs, "compression_level")) {
    output.compression_level =
        strict_int32(kwargs["compression_level"], "Output.compression_level");
  }
  if (has_key(kwargs, "save_namelist_copy")) {
    output.save_namelist_copy =
        strict_bool(kwargs["save_namelist_copy"], "Output.save_namelist_copy");
  }
  if (has_key(kwargs, "save_frozen_config")) {
    output.save_frozen_config =
        strict_bool(kwargs["save_frozen_config"], "Output.save_frozen_config");
  }
  if (has_key(kwargs, "plot_fields")) {
    const py::handle fields_obj = kwargs["plot_fields"];
    if (!py::isinstance<py::sequence>(fields_obj) || py::isinstance<py::str>(fields_obj)) {
      throw_value_type_error("Output.plot_fields", "list[str]", fields_obj);
    }
    output.plot_fields.clear();
    for (const py::handle field : py::reinterpret_borrow<py::sequence>(fields_obj)) {
      output.plot_fields.push_back(strict_string(field, "Output.plot_fields[]"));
    }
  }
}

void Builder::set_diagnostics(py::dict kwargs) {
  mark_block_called(Block::Diagnostics);
  enforce_known_keys(kwargs, "Diagnostics",
                     {"enabled", "every", "energy_budget", "areal_density", "sphericity",
                      "shell", "laser_pattern", "mc_stats", "fleck_diag",
                      "overshoot_monitor", "per_operator_radial_fourier_enabled",
                      "radial_fourier_window_t_start_s",
                      "radial_fourier_window_t_end_s", "radial_fourier_max_mode",
                      "per_operator_radial_fourier_complex_enabled",
                      "per_operator_radial_fourier_complex_m_targets",
                      "per_operator_radial_fourier_complex_j_targets",
                      "per_operator_radial_fourier_complex_fields"});

  auto& diagnostics = config.diagnostics;
  if (has_key(kwargs, "enabled")) {
    diagnostics.enabled = strict_bool(kwargs["enabled"], "Diagnostics.enabled");
  }
  if (has_key(kwargs, "every")) {
    diagnostics.every =
        strict_int32(kwargs["every"], "Diagnostics.every");
    ensure_int_ge(diagnostics.every, 1, "Diagnostics.every");
  }
  if (has_key(kwargs, "overshoot_monitor")) {
    diagnostics.overshoot_monitor =
        strict_bool(kwargs["overshoot_monitor"], "Diagnostics.overshoot_monitor");
  }
  if (has_key(kwargs, "per_operator_radial_fourier_enabled")) {
    diagnostics.per_operator_radial_fourier_enabled = strict_bool(
        kwargs["per_operator_radial_fourier_enabled"],
        "Diagnostics.per_operator_radial_fourier_enabled");
  }
  if (has_key(kwargs, "radial_fourier_window_t_start_s")) {
    diagnostics.radial_fourier_window_t_start_s = numeric_as_double(
        kwargs["radial_fourier_window_t_start_s"],
        "Diagnostics.radial_fourier_window_t_start_s");
  }
  if (has_key(kwargs, "radial_fourier_window_t_end_s")) {
    diagnostics.radial_fourier_window_t_end_s = numeric_as_double(
        kwargs["radial_fourier_window_t_end_s"],
        "Diagnostics.radial_fourier_window_t_end_s");
  }
  if (has_key(kwargs, "radial_fourier_max_mode")) {
    diagnostics.radial_fourier_max_mode =
        strict_int32(kwargs["radial_fourier_max_mode"],
                     "Diagnostics.radial_fourier_max_mode");
    ensure_int_ge(diagnostics.radial_fourier_max_mode, -1,
                  "Diagnostics.radial_fourier_max_mode");
  }
  if (has_key(kwargs, "per_operator_radial_fourier_complex_enabled")) {
    diagnostics.per_operator_radial_fourier_complex_enabled = strict_bool(
        kwargs["per_operator_radial_fourier_complex_enabled"],
        "Diagnostics.per_operator_radial_fourier_complex_enabled");
  }
  if (has_key(kwargs, "per_operator_radial_fourier_complex_m_targets")) {
    diagnostics.per_operator_radial_fourier_complex_m_targets = strict_int_vector(
        kwargs["per_operator_radial_fourier_complex_m_targets"],
        "Diagnostics.per_operator_radial_fourier_complex_m_targets");
    for (const int m : diagnostics.per_operator_radial_fourier_complex_m_targets) {
      ensure_int_ge(m, 0,
                    "Diagnostics.per_operator_radial_fourier_complex_m_targets[]");
    }
  }
  if (has_key(kwargs, "per_operator_radial_fourier_complex_j_targets")) {
    diagnostics.per_operator_radial_fourier_complex_j_targets = strict_int_vector(
        kwargs["per_operator_radial_fourier_complex_j_targets"],
        "Diagnostics.per_operator_radial_fourier_complex_j_targets");
    for (const int j : diagnostics.per_operator_radial_fourier_complex_j_targets) {
      ensure_int_ge(j, 0,
                    "Diagnostics.per_operator_radial_fourier_complex_j_targets[]");
    }
  }
  if (has_key(kwargs, "per_operator_radial_fourier_complex_fields")) {
    diagnostics.per_operator_radial_fourier_complex_fields = strict_string_vector(
        kwargs["per_operator_radial_fourier_complex_fields"],
        "Diagnostics.per_operator_radial_fourier_complex_fields");
  }
  if (diagnostics.radial_fourier_window_t_end_s <
      diagnostics.radial_fourier_window_t_start_s) {
    throw ValueError(
        "Diagnostics.radial_fourier_window_t_end_s must be >= "
        "radial_fourier_window_t_start_s");
  }
  if (has_key(kwargs, "shell")) {
    warn_ignored_key("Diagnostics.shell");
  }

  if (has_key(kwargs, "energy_budget")) {
    const py::handle energy_obj = kwargs["energy_budget"];
    if (!py::isinstance<py::dict>(energy_obj)) {
      throw_value_type_error("Diagnostics.energy_budget", "dict", energy_obj);
    }
    const py::dict energy = py::reinterpret_borrow<py::dict>(energy_obj);
    enforce_known_keys(energy, "Diagnostics.energy_budget",
                       {"enabled", "warn_threshold"});
    if (has_key(energy, "enabled")) {
      diagnostics.energy_budget.enabled =
          strict_bool(energy["enabled"], "Diagnostics.energy_budget.enabled");
    }
    if (has_key(energy, "warn_threshold")) {
      diagnostics.energy_budget.warn_threshold = numeric_as_double(
          energy["warn_threshold"], "Diagnostics.energy_budget.warn_threshold");
    }
  }
  if (has_key(kwargs, "areal_density")) {
    const py::handle areal_obj = kwargs["areal_density"];
    if (!py::isinstance<py::dict>(areal_obj)) {
      throw_value_type_error("Diagnostics.areal_density", "dict", areal_obj);
    }
    const py::dict areal = py::reinterpret_borrow<py::dict>(areal_obj);
    enforce_known_keys(areal, "Diagnostics.areal_density",
                       {"enabled", "r_range", "angles_deg"});
    if (has_key(areal, "enabled")) {
      diagnostics.areal_density.enabled =
          strict_bool(areal["enabled"], "Diagnostics.areal_density.enabled");
    }
    if (has_key(areal, "r_range")) {
      diagnostics.areal_density.r_range =
          strict_string(areal["r_range"], "Diagnostics.areal_density.r_range");
    }
    if (has_key(areal, "angles_deg")) {
      diagnostics.areal_density.angles_deg = strict_double_vector(
          areal["angles_deg"], "Diagnostics.areal_density.angles_deg");
    }
  }
  if (has_key(kwargs, "sphericity")) {
    const py::handle sph_obj = kwargs["sphericity"];
    if (!py::isinstance<py::dict>(sph_obj)) {
      throw_value_type_error("Diagnostics.sphericity", "dict", sph_obj);
    }
    const py::dict sph = py::reinterpret_borrow<py::dict>(sph_obj);
    enforce_known_keys(sph, "Diagnostics.sphericity",
                       {"enabled", "surface", "rho_threshold", "modes"});
    if (has_key(sph, "enabled")) {
      diagnostics.sphericity.enabled =
          strict_bool(sph["enabled"], "Diagnostics.sphericity.enabled");
    }
    if (has_key(sph, "surface")) {
      diagnostics.sphericity.surface =
          strict_string(sph["surface"], "Diagnostics.sphericity.surface");
    }
    if (has_key(sph, "rho_threshold")) {
      diagnostics.sphericity.rho_threshold = numeric_as_double(
          sph["rho_threshold"], "Diagnostics.sphericity.rho_threshold");
    }
    if (has_key(sph, "modes")) {
      diagnostics.sphericity.modes =
          strict_int_vector(sph["modes"], "Diagnostics.sphericity.modes");
    }
  }
  if (has_key(kwargs, "laser_pattern")) {
    const py::handle laser_obj = kwargs["laser_pattern"];
    if (!py::isinstance<py::dict>(laser_obj)) {
      throw_value_type_error("Diagnostics.laser_pattern", "dict", laser_obj);
    }
    const py::dict laser = py::reinterpret_borrow<py::dict>(laser_obj);
    enforce_known_keys(laser, "Diagnostics.laser_pattern",
                       {"enabled", "absorbed_power_profile", "critical_surface",
                        "per_beam"});
    if (has_key(laser, "enabled")) {
      diagnostics.laser_pattern.enabled =
          strict_bool(laser["enabled"], "Diagnostics.laser_pattern.enabled");
    }
    if (has_key(laser, "absorbed_power_profile")) {
      diagnostics.laser_pattern.absorbed_power_profile = strict_bool(
          laser["absorbed_power_profile"],
          "Diagnostics.laser_pattern.absorbed_power_profile");
    }
    if (has_key(laser, "critical_surface")) {
      diagnostics.laser_pattern.critical_surface = strict_bool(
          laser["critical_surface"],
          "Diagnostics.laser_pattern.critical_surface");
    }
    if (has_key(laser, "per_beam")) {
      diagnostics.laser_pattern.per_beam =
          strict_bool(laser["per_beam"], "Diagnostics.laser_pattern.per_beam");
    }
  }
  if (has_key(kwargs, "mc_stats")) {
    const py::handle mc_obj = kwargs["mc_stats"];
    if (!py::isinstance<py::dict>(mc_obj)) {
      throw_value_type_error("Diagnostics.mc_stats", "dict", mc_obj);
    }
    const py::dict mc = py::reinterpret_borrow<py::dict>(mc_obj);
    enforce_known_keys(mc, "Diagnostics.mc_stats",
                       {"enabled", "particle_counts", "weight_stats",
                        "cell_particle_density", "ddmc_fraction"});
    if (has_key(mc, "enabled")) {
      diagnostics.mc_stats.enabled =
          strict_bool(mc["enabled"], "Diagnostics.mc_stats.enabled");
    }
    if (has_key(mc, "particle_counts")) {
      diagnostics.mc_stats.particle_counts =
          strict_bool(mc["particle_counts"], "Diagnostics.mc_stats.particle_counts");
    }
    if (has_key(mc, "weight_stats")) {
      diagnostics.mc_stats.weight_stats =
          strict_bool(mc["weight_stats"], "Diagnostics.mc_stats.weight_stats");
    }
    if (has_key(mc, "cell_particle_density")) {
      diagnostics.mc_stats.cell_particle_density = strict_bool(
          mc["cell_particle_density"], "Diagnostics.mc_stats.cell_particle_density");
    }
    if (has_key(mc, "ddmc_fraction")) {
      diagnostics.mc_stats.ddmc_fraction =
          strict_bool(mc["ddmc_fraction"], "Diagnostics.mc_stats.ddmc_fraction");
    }
  }
  if (has_key(kwargs, "fleck_diag")) {
    const py::handle fleck_obj = kwargs["fleck_diag"];
    if (!py::isinstance<py::dict>(fleck_obj)) {
      throw_value_type_error("Diagnostics.fleck_diag", "dict", fleck_obj);
    }
    const py::dict fleck = py::reinterpret_borrow<py::dict>(fleck_obj);
    enforce_known_keys(fleck, "Diagnostics.fleck_diag",
                       {"enabled", "every", "cells", "r_min_cm", "r_max_cm"});
    if (has_key(fleck, "enabled")) {
      diagnostics.fleck_diag.enabled =
          strict_bool(fleck["enabled"], "Diagnostics.fleck_diag.enabled");
    }
    if (has_key(fleck, "every")) {
      diagnostics.fleck_diag.every =
          strict_int32(fleck["every"], "Diagnostics.fleck_diag.every");
      ensure_int_ge(diagnostics.fleck_diag.every, 1, "Diagnostics.fleck_diag.every");
    }
    if (has_key(fleck, "cells")) {
      diagnostics.fleck_diag.cells =
          strict_int_vector(fleck["cells"], "Diagnostics.fleck_diag.cells");
      for (const int cell : diagnostics.fleck_diag.cells) {
        if (cell < 0) {
          throw ValueError("Diagnostics.fleck_diag.cells entries must be >= 0");
        }
      }
    }
    if (has_key(fleck, "r_min_cm")) {
      diagnostics.fleck_diag.r_min_cm = numeric_as_double(
          fleck["r_min_cm"], "Diagnostics.fleck_diag.r_min_cm");
    }
    if (has_key(fleck, "r_max_cm")) {
      diagnostics.fleck_diag.r_max_cm = numeric_as_double(
          fleck["r_max_cm"], "Diagnostics.fleck_diag.r_max_cm");
    }

    const bool has_r_min = diagnostics.fleck_diag.r_min_cm >= 0.0;
    const bool has_r_max = diagnostics.fleck_diag.r_max_cm >= 0.0;
    if (has_r_min != has_r_max) {
      throw ConfigError(
          "Diagnostics.fleck_diag requires both r_min_cm and r_max_cm when using radius selection");
    }
    if (has_r_min && diagnostics.fleck_diag.r_max_cm < diagnostics.fleck_diag.r_min_cm) {
      throw ValueError("Diagnostics.fleck_diag.r_max_cm must be >= r_min_cm");
    }
  }
}

void Builder::set_parallel(py::dict kwargs) {
  mark_block_called(Block::Parallel);
  enforce_known_keys(kwargs, "Parallel",
                     {"decomposition", "halo", "migration", "laser_parallel",
                      "particle_balance", "reproducibility", "gpu_optimization"});

  auto& parallel = config.parallel;
  if (has_key(kwargs, "laser_parallel")) {
    warn_ignored_key("Parallel.laser_parallel");
  }
  if (has_key(kwargs, "particle_balance")) {
    warn_ignored_key("Parallel.particle_balance");
  }
  if (has_key(kwargs, "reproducibility")) {
    warn_ignored_key("Parallel.reproducibility");
  }
  if (has_key(kwargs, "gpu_optimization")) {
    warn_ignored_key("Parallel.gpu_optimization");
  }
  if (has_key(kwargs, "decomposition")) {
    const py::handle decomposition_obj = kwargs["decomposition"];
    if (!py::isinstance<py::dict>(decomposition_obj)) {
      throw_value_type_error("Parallel.decomposition", "dict", decomposition_obj);
    }
    const py::dict decomposition = py::reinterpret_borrow<py::dict>(decomposition_obj);
    enforce_known_keys(decomposition, "Parallel.decomposition",
                       {"method", "dims", "min_cells_per_rank"});
    if (has_key(decomposition, "method")) {
      parallel.decomposition.method =
          strict_string(decomposition["method"], "Parallel.decomposition.method");
    }
    if (has_key(decomposition, "dims")) {
      parallel.decomposition.dims =
          strict_int_vector(decomposition["dims"], "Parallel.decomposition.dims");
    }
    if (has_key(decomposition, "min_cells_per_rank")) {
      parallel.decomposition.min_cells_per_rank =
          strict_int32(decomposition["min_cells_per_rank"],
                       "Parallel.decomposition.min_cells_per_rank");
    }
  }
  if (has_key(kwargs, "halo")) {
    const py::handle halo_obj = kwargs["halo"];
    if (!py::isinstance<py::dict>(halo_obj)) {
      throw_value_type_error("Parallel.halo", "dict", halo_obj);
    }
    const py::dict halo = py::reinterpret_borrow<py::dict>(halo_obj);
    enforce_known_keys(halo, "Parallel.halo", {"gpu_aware_mpi", "ghost_layers"});
    if (has_key(halo, "gpu_aware_mpi")) {
      parallel.halo.gpu_aware_mpi =
          strict_string(halo["gpu_aware_mpi"], "Parallel.halo.gpu_aware_mpi");
    }
    if (has_key(halo, "ghost_layers")) {
      parallel.halo.ghost_layers =
          strict_int32(halo["ghost_layers"], "Parallel.halo.ghost_layers");
    }
  }
  if (has_key(kwargs, "migration")) {
    const py::handle migration_obj = kwargs["migration"];
    if (!py::isinstance<py::dict>(migration_obj)) {
      throw_value_type_error("Parallel.migration", "dict", migration_obj);
    }
    const py::dict migration = py::reinterpret_borrow<py::dict>(migration_obj);
    enforce_known_keys(migration, "Parallel.migration",
                       {"method", "max_substeps", "emigrant_threshold",
                        "initial_capacity", "growth_factor"});
    if (has_key(migration, "method")) {
      parallel.migration.method =
          strict_string(migration["method"], "Parallel.migration.method");
    }
    if (has_key(migration, "max_substeps")) {
      parallel.migration.max_substeps =
          strict_int32(migration["max_substeps"], "Parallel.migration.max_substeps");
    }
    if (has_key(migration, "emigrant_threshold")) {
      parallel.migration.emigrant_threshold = strict_int32(
          migration["emigrant_threshold"], "Parallel.migration.emigrant_threshold");
    }
    if (has_key(migration, "initial_capacity")) {
      parallel.migration.initial_capacity = strict_int32(
          migration["initial_capacity"], "Parallel.migration.initial_capacity");
    }
    if (has_key(migration, "growth_factor")) {
      parallel.migration.growth_factor = numeric_as_double(
          migration["growth_factor"], "Parallel.migration.growth_factor");
    }
  }
}

void Builder::set_burn(py::dict kwargs) {
  mark_block_called(Block::Burn);
  enforce_known_keys(kwargs, "Burn",
                     {"enabled", "fuels", "scheme", "partition", "screening",
                      "diffusion_groups", "diffusion_E_min_keV",
                      "mc_particles_per_cell",
                      "fuel_materials", "x_D", "x_T", "x_He3", "T_floor_keV",
                      "explicit_source_limit", "eps_deplete", "subcycle_max",
                      "vf_threshold", "neutron_heating", "neutron_heating_n_mu"});

  auto& burn = config.burn;
  const auto strict_list_or_tuple_string_vector =
      [](const py::handle value, std::string_view path) {
        if (!py::isinstance<py::list>(value) && !py::isinstance<py::tuple>(value)) {
          throw_value_type_error(path, "list[str]", value);
        }
        return strict_string_vector(value, path);
      };

  if (has_key(kwargs, "enabled")) {
    burn.enabled = strict_bool(kwargs["enabled"], "Burn.enabled");
  }
  if (has_key(kwargs, "fuels")) {
    burn.fuels = strict_list_or_tuple_string_vector(kwargs["fuels"], "Burn.fuels");
    if (burn.fuels.empty()) {
      throw ValueError("Burn.fuels must not be empty");
    }
    std::set<std::string> seen;
    for (const auto& fuel : burn.fuels) {
      if (fuel != "DT" && fuel != "DD" && fuel != "D3He") {
        throw ValueError("Burn.fuels entries must be \"DT\", \"DD\", or \"D3He\"");
      }
      if (!seen.insert(fuel).second) {
        throw ValueError("Burn.fuels must not contain duplicates");
      }
    }
  }
  if (has_key(kwargs, "scheme")) {
    burn.scheme = strict_string(kwargs["scheme"], "Burn.scheme");
    if (burn.scheme != "fraley" && burn.scheme != "diffusion" &&
        burn.scheme != "mc" && burn.scheme != "local") {
      throw ValueError(
          "Burn.scheme must be \"fraley\", \"diffusion\", \"mc\", or \"local\"");
    }
  }
  if (has_key(kwargs, "diffusion_groups")) {
    burn.diffusion_groups =
        strict_int32(kwargs["diffusion_groups"], "Burn.diffusion_groups");
    if (burn.diffusion_groups < 4 || burn.diffusion_groups > 512) {
      throw ValueError("Burn.diffusion_groups must be in [4, 512]");
    }
  }
  if (has_key(kwargs, "diffusion_E_min_keV")) {
    burn.diffusion_E_min_keV =
        numeric_as_double(kwargs["diffusion_E_min_keV"],
                          "Burn.diffusion_E_min_keV");
    if (!(burn.diffusion_E_min_keV > 1.0) ||
        burn.diffusion_E_min_keV > 100.0) {
      throw ValueError("Burn.diffusion_E_min_keV must be in (1.0, 100.0]");
    }
  }
  if (has_key(kwargs, "mc_particles_per_cell")) {
    burn.mc_particles_per_cell =
        strict_int32(kwargs["mc_particles_per_cell"],
                     "Burn.mc_particles_per_cell");
    if (burn.mc_particles_per_cell < 1 ||
        burn.mc_particles_per_cell > 4096) {
      throw ValueError("Burn.mc_particles_per_cell must be in [1, 4096]");
    }
  }
  if (has_key(kwargs, "partition")) {
    burn.partition = strict_string(kwargs["partition"], "Burn.partition");
    if (burn.partition != "li_petrasso" && burn.partition != "fraley") {
      throw ValueError("Burn.partition must be \"li_petrasso\" or \"fraley\"");
    }
  }
  if (has_key(kwargs, "screening")) {
    burn.screening = strict_string(kwargs["screening"], "Burn.screening");
    if (burn.screening != "none" && burn.screening != "salpeter" &&
        burn.screening != "chugunov_dewitt") {
      throw ValueError(
          "Burn.screening must be \"none\", \"salpeter\", or \"chugunov_dewitt\"");
    }
  }
  if (has_key(kwargs, "fuel_materials")) {
    burn.fuel_materials =
        strict_list_or_tuple_string_vector(kwargs["fuel_materials"], "Burn.fuel_materials");
    if (burn.fuel_materials.empty()) {
      throw ValueError("Burn.fuel_materials must not be empty");
    }
    for (const auto& material_name : burn.fuel_materials) {
      if (material_name.empty()) {
        throw ValueError("Burn.fuel_materials entries must be non-empty strings");
      }
    }
  }
  if (has_key(kwargs, "x_D")) {
    burn.x_D = numeric_as_double(kwargs["x_D"], "Burn.x_D");
    if (burn.x_D < 0.0) {
      throw ValueError("Burn.x_D must be >= 0");
    }
  }
  if (has_key(kwargs, "x_T")) {
    burn.x_T = numeric_as_double(kwargs["x_T"], "Burn.x_T");
    if (burn.x_T < 0.0) {
      throw ValueError("Burn.x_T must be >= 0");
    }
  }
  if (has_key(kwargs, "x_He3")) {
    burn.x_He3 = numeric_as_double(kwargs["x_He3"], "Burn.x_He3");
    if (burn.x_He3 < 0.0) {
      throw ValueError("Burn.x_He3 must be >= 0");
    }
  }
  if (has_key(kwargs, "T_floor_keV")) {
    burn.T_floor_keV = numeric_as_double(kwargs["T_floor_keV"], "Burn.T_floor_keV");
    if (!(burn.T_floor_keV > 0.0)) {
      throw ValueError("Burn.T_floor_keV must be > 0");
    }
  }
  if (has_key(kwargs, "explicit_source_limit")) {
    burn.explicit_source_limit =
        numeric_as_double(kwargs["explicit_source_limit"], "Burn.explicit_source_limit");
    if (!(burn.explicit_source_limit > 0.0) || burn.explicit_source_limit > 1.0) {
      throw ValueError("Burn.explicit_source_limit must be in (0, 1]");
    }
  }
  if (has_key(kwargs, "eps_deplete")) {
    burn.eps_deplete = numeric_as_double(kwargs["eps_deplete"], "Burn.eps_deplete");
    if (!(burn.eps_deplete > 0.0) || burn.eps_deplete > 0.5) {
      throw ValueError("Burn.eps_deplete must be in (0, 0.5]");
    }
  }
  if (has_key(kwargs, "subcycle_max")) {
    burn.subcycle_max = strict_int32(kwargs["subcycle_max"], "Burn.subcycle_max");
    if (burn.subcycle_max < 1 || burn.subcycle_max > 4096) {
      throw ValueError("Burn.subcycle_max must be in [1, 4096]");
    }
  }
  if (has_key(kwargs, "vf_threshold")) {
    burn.vf_threshold = numeric_as_double(kwargs["vf_threshold"], "Burn.vf_threshold");
    if (!(burn.vf_threshold > 0.0) || burn.vf_threshold >= 1.0) {
      throw ValueError("Burn.vf_threshold must be in (0, 1)");
    }
  }
  if (has_key(kwargs, "neutron_heating")) {
    burn.neutron_heating =
        strict_bool(kwargs["neutron_heating"], "Burn.neutron_heating");
  }
  if (has_key(kwargs, "neutron_heating_n_mu")) {
    burn.neutron_heating_n_mu = strict_int32(kwargs["neutron_heating_n_mu"],
                                             "Burn.neutron_heating_n_mu");
    if (burn.neutron_heating_n_mu < 2 || burn.neutron_heating_n_mu > 64 ||
        (burn.neutron_heating_n_mu % 2) != 0) {
      throw ValueError("Burn.neutron_heating_n_mu must be even and in [2, 64]");
    }
  }
  if (burn.partition == "fraley" && burn.neutron_heating) {
    throw ConfigError(
        "Burn.partition=\"fraley\" is the Fraley DT-alpha local-deposition "
        "fit; it cannot partition neutron elastic recoils (D/T). Use "
        "partition=\"li_petrasso\" with neutron_heating=True.");
  }

  const double x_sum = burn.x_D + burn.x_T + burn.x_He3;
  if (std::abs(x_sum - 1.0) > 1.0e-6) {
    std::ostringstream oss;
    oss << "Burn.x_D + Burn.x_T + Burn.x_He3 must sum to 1 (got " << x_sum
        << "); set all three explicitly for non-default mixes";
    throw ValueError(oss.str());
  }
}

void Builder::validate() {
  const auto required_block = [this](const Block block, std::string_view block_name) {
    if (!blocks_called.test(static_cast<std::size_t>(block))) {
      throw ConfigError(std::string("Required block not provided: ") +
                        std::string(block_name));
    }
  };
  required_block(Block::Main, "Main");
  required_block(Block::Mesh, "Mesh");
  required_block(Block::Materials, "Materials");
  required_block(Block::Geometry, "Geometry");

  auto& main = config.main;
  auto& mesh = config.mesh;
  auto& materials = config.materials;
  auto& geometry = config.geometry;
  auto& radiation = config.radiation;
  auto& laser = config.laser;
  auto& numerics = config.numerics;
  auto& parallel = config.parallel;
  auto& burn = config.burn;
  const bool radiation_block_called =
      blocks_called.test(static_cast<std::size_t>(Block::Radiation));
  if (!radiation_block_called && radiation.mode == RadiationMode::MultigroupDiffusion) {
    radiation.imc.enabled = false;
    radiation.ddmc.enabled = false;
  }

  if (!main_name_explicit && main.name == "unnamed") {
    const std::filesystem::path source_path(config.meta.namelist_source_path);
    const std::string stem = source_path.stem().string();
    if (!stem.empty()) {
      main.name = stem;
    }
  }
  if (main.name.empty()) {
    throw ConfigError("Main.name must not be empty");
  }
  if (!is_dimension(main.dimension)) {
    throw ConfigError("Main.dimension must be \"1D_SPH\", \"1D_CYL\", or \"2D_RZ\"");
  }
  if (numerics.hydro.pressure_drive_perturbation.enabled &&
      main.dimension != "2D_RZ") {
    throw ConfigError(
        "pressure_drive_perturbation requires a 2D_RZ pressure drive");
  }
  if (numerics.hydro.pressure_drive_perturbation.enabled &&
      numerics.hydro.ring7_quotient_enabled) {
    throw ConfigError(
        "pressure_drive_perturbation is not supported with the ring7 seam quotient");
  }
  if (mesh.shell_polar_cap_dendrite && laser.enabled) {
    throw ConfigError(
        "Laser.enabled=true is not supported with "
        "Mesh.shell_polar_cap_dendrite=true (laser+shellcap is out of v1 scope)");
  }
  if (mesh.shell_polar_cap_dendrite &&
      numerics.ale.pole_axis_bbsw_enabled) {
    throw ConfigError(
        "Numerics.ale.pole_axis_bbsw_enabled=true is not supported with "
        "Mesh.shell_polar_cap_dendrite=true (pending shell-chain generalization)");
  }
  if (mesh.shell_polar_cap_dendrite &&
      numerics.hydro.ring7_quotient_enabled) {
    throw ConfigError(
        "Numerics.hydro.ring7_quotient_enabled=true is not supported with "
        "Mesh.shell_polar_cap_dendrite=true (pending shell-chain generalization)");
  }
  if (!is_temperature_model(main.temperature_model)) {
    throw ConfigError("Main.temperature_model must be \"1T\", \"2T\", or \"auto\"");
  }
  main.two_temperature = (main.temperature_model == "2T");
  main.dim = (main.dimension == "2D_RZ") ? 2 : 1;
  // 2026-08-03 AV modernization: 1D_SPH default is limited
  // CSW98; explicit av_type (including "vnr") is always honored, and
  // frozen configs are unaffected (av_type is always emitted).
  if (!numerics.hydro.av_type_explicit && main.dimension == "1D_SPH") {
    numerics.hydro.av_type = "csw";
  }
  if (numerics.conduction.nonlocal_model != "none" &&
      numerics.conduction.nonlocal_model != "snb") {
    throw ConfigError(
        "Numerics.conduction.nonlocal_model must be \"none\" or \"snb\"");
  }
  if (numerics.conduction.nonlocal_model == "snb") {
    if (main.dimension != "1D_SPH" && main.dimension != "2D_RZ") {
      throw ConfigError(
          "Numerics.conduction.nonlocal_model=\"snb\" is supported only for 1D_SPH"
          " (planar/cylindrical/spherical via Mesh.geometry_1d) or 2D_RZ");
    }
    if (numerics.conduction.solver != "sts") {
      throw ConfigError(
          "Numerics.conduction.nonlocal_model=\"snb\" requires"
          " Numerics.conduction.solver=\"sts\"");
    }
    if (!main.two_temperature) {
      throw ConfigError(
          "Numerics.conduction.nonlocal_model=\"snb\" requires Main.temperature_model=\"2T\"");
    }
    if (!numerics.conduction.enabled) {
      throw ConfigError(
          "Numerics.conduction.nonlocal_model=\"snb\" requires Numerics.conduction.enabled=True");
    }
    if (numerics.materials.per_material_conservation_enabled) {
      throw ConfigError(
          "Numerics.conduction.nonlocal_model=\"snb\" does not support"
          " Numerics.materials.per_material_conservation_enabled=True (v1)");
    }
    if (numerics.conduction.snb_n_groups < 2) {
      throw ConfigError("Numerics.conduction.snb_n_groups must be >= 2");
    }
    if (!(numerics.conduction.snb_E_max_over_Te > 1.0)) {
      throw ConfigError("Numerics.conduction.snb_E_max_over_Te must be > 1");
    }
    if (numerics.conduction.snb_mfp != "geometric_r2" &&
        numerics.conduction.snb_mfp != "original") {
      throw ConfigError(
          "Numerics.conduction.snb_mfp must be \"geometric_r2\" or \"original\"");
    }
    if (numerics.conduction.snb_efield != "none" &&
        numerics.conduction.snb_efield != "local") {
      throw ConfigError(
          "Numerics.conduction.snb_efield must be \"none\" or \"local\"");
    }
    if (main.dimension == "2D_RZ" && numerics.conduction.snb_efield != "none") {
      throw ConfigError(
          "Numerics.conduction.snb_efield=\"local\" is not available in 2D_RZ"
          " v1 (fail-closed; docs/design/2d_snb_port_spec.md §2)");
    }
    if (numerics.conduction.snb_picard_max_iters < 2) {
      throw ConfigError("Numerics.conduction.snb_picard_max_iters must be >= 2");
    }
    if (!(numerics.conduction.snb_picard_rtol > 0.0)) {
      throw ConfigError("Numerics.conduction.snb_picard_rtol must be > 0");
    }
  }
  if ((is_dimension_1d(main.dimension) || main.dimension == "2D_RZ") &&
      radiation.enabled) {
    const bool mode_ok =
        radiation.mode == RadiationMode::MultigroupDiffusion ||
        radiation.mode == RadiationMode::SnTransport;
    if (!mode_ok) {
      throw ConfigError(
          "1D_SPH and 2D_RZ production radiation support only "
          "Radiation.mode=\"multigroup_diffusion\" or \"sn_transport\". "
          "Other modes (\"imc_ddmc\" including IMC/DDMC/HOLO/difference "
          "formulations) are frozen.");
    }
  }
  if (laser.enabled) {
    const auto& ports = laser.port_configuration.ports;
    if (!ports.empty()) {
      if (laser.port_configuration.normalization != "sum_weights_one") {
        throw ConfigError(
            "Laser.port_configuration.normalization must be "
            "\"sum_weights_one\"");
      }
      if (ports.size() < 1 || ports.size() > 192) {
        throw ConfigError(
            "Laser.port_configuration.ports must contain between 1 and 192 ports");
      }
      std::unordered_set<int> port_ids;
      double power_weight_sum = 0.0;
      for (std::size_t i = 0; i < ports.size(); ++i) {
        const auto& port = ports[i];
        const std::string path =
            "Laser.port_configuration.ports[" + std::to_string(i) + "]";
        if (port.port_id < 0) {
          throw ConfigError(path + ".port_id must be >= 0");
        }
        if (!port_ids.insert(port.port_id).second) {
          throw ConfigError(
              "Laser.port_configuration.ports contains duplicate port_id " +
              std::to_string(port.port_id));
        }
        if (port.direction.size() != 3) {
          throw ConfigError(path + ".direction must contain exactly 3 values");
        }
        const double direction_norm =
            std::sqrt(port.direction[0] * port.direction[0] +
                      port.direction[1] * port.direction[1] +
                      port.direction[2] * port.direction[2]);
        if (!(direction_norm > 0.0)) {
          throw ConfigError(path + ".direction must have nonzero norm");
        }
        if (!(port.power_weight > 0.0)) {
          throw ConfigError(path + ".power_weight must be > 0");
        }
        power_weight_sum += port.power_weight;
        if (port.polarization != "unpolarized") {
          throw ConfigError(path +
                            ".polarization must be \"unpolarized\"");
        }
      }
      if (std::abs(power_weight_sum - 1.0) > 1.0e-6) {
        std::ostringstream oss;
        oss << std::setprecision(17)
            << "Laser.port_configuration power_weight sum must equal 1 "
               "within 1e-6, got "
            << power_weight_sum;
        throw ConfigError(oss.str());
      }
    }

    if (laser.cbet.geometry_mode != "legacy" &&
        laser.cbet.geometry_mode != "port_section") {
      throw ConfigError(
          "Laser.cbet.geometry_mode must be \"legacy\" or \"port_section\", got \"" +
          laser.cbet.geometry_mode + "\"");
    }
    if (laser.cbet.geometry_mode == "port_section") {
      if (main.dimension != "1D_SPH") {
        throw ConfigError(
            "Laser.cbet.geometry_mode=\"port_section\" requires "
            "Main.dimension=\"1D_SPH\"");
      }
      if (!laser.cbet.enable) {
        throw ConfigError(
            "Laser.cbet.geometry_mode=\"port_section\" requires "
            "Laser.cbet.enable=True");
      }
      if (ports.empty()) {
        throw ConfigError(
            "Laser.cbet.geometry_mode=\"port_section\" requires "
            "Laser.port_configuration.ports");
      }
      if (laser.beams.size() != 1) {
        throw ConfigError(
            "Laser.cbet.geometry_mode=\"port_section\" requires exactly one "
            "Laser.beams prototype");
      }
      if (laser.cbet.n_section_phi < 4 ||
          laser.cbet.n_section_phi > 64) {
        throw ConfigError(
            "Laser.cbet.n_section_phi must be in [4, 64] when "
            "Laser.cbet.geometry_mode=\"port_section\"");
      }
    }
  }
  if (laser.enabled && laser.cbet.enable) {
    const bool cbet_ok_1d = (main.dimension == "1D_SPH" && laser.mode == "raytrace_2d");
    const bool cbet_ok_2d = (main.dimension == "2D_RZ" && laser.mode == "raytrace_3d");
    if (!cbet_ok_1d && !cbet_ok_2d) {
      throw ConfigError(
          "Laser.cbet.enable=True requires Main.dimension=\"1D_SPH\" with "
          "Laser.mode=\"raytrace_2d\", or Main.dimension=\"2D_RZ\" with "
          "Laser.mode=\"raytrace_3d\"");
    }
  }
  if (burn.enabled) {
    if (main.dimension == "1D_SPH") {
      if (mesh.geometry_1d != "spherical") {
        throw ConfigError(
            "Burn.enabled=True requires Mesh.geometry_1d=\"spherical\" (v1)");
      }
      if (burn.scheme == "local") {
        throw ConfigError(
            "Burn.scheme=\"local\" is supported only for Main.dimension=\"2D_RZ\"; "
            "1D_SPH supports \"fraley\", \"diffusion\", or \"mc\"");
      }
    } else if (main.dimension == "2D_RZ") {
      if (burn.scheme != "local" && burn.scheme != "diffusion") {
        throw ConfigError(
            "Burn.scheme=\"" + burn.scheme +
            "\" is not supported for Main.dimension=\"2D_RZ\"; 2D supports "
            "\"local\" or \"diffusion\" (\"fraley\" is 1D-spherical-only; "
            "\"mc\" is not yet ported)");
      }
      if (numerics.ale.enabled) {
        if (!numerics.ale.conservative_remap_enabled) {
          throw ConfigError(
              "Burn.enabled=True with ALE requires "
              "Numerics.ale.conservative_remap_enabled=True (2D species "
              "transport is wired on the CSR conservative remap only)");
        }
        if (numerics.materials.per_material_conservation_enabled) {
          throw ConfigError(
              "Burn.enabled=True with ALE does not support "
              "Numerics.materials.per_material_conservation_enabled=True "
              "(the PLIC per-material remap is not species-wired; v1)");
        }
        if (numerics.hydro.total_energy_remap_2d_rz) {
          throw ConfigError(
              "Burn.enabled=True with ALE does not support "
              "Numerics.hydro.total_energy_remap_2d_rz=True (the total-energy "
              "remap kernels are not species-wired; v1)");
        }
      }
      if (numerics.ale.force_rezone_every_n_steps > 0) {
        throw ConfigError(
            "Burn.enabled=True with "
            "Numerics.ale.force_rezone_every_n_steps>0 engages the "
            "rezone/PLIC remap which is not species-wired; fail-closed "
            "refusal");
      }
      if (numerics.ale.reference_barrier_enabled) {
        throw ConfigError(
            "Burn.enabled=True with "
            "Numerics.ale.reference_barrier_enabled=True engages the "
            "rezone/PLIC remap which is not species-wired; fail-closed "
            "refusal");
      }
      if (numerics.ale.axis_band_managed_remap_enabled) {
        throw ConfigError(
            "Burn.enabled=True with "
            "Numerics.ale.axis_band_managed_remap_enabled=True is not yet "
            "supported for 2D_RZ (burn species remap lands in a later wave; "
            "fail-closed refusal)");
      }
      if (mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK) {
        throw ConfigError(
            "Burn.enabled=True requires the single_block 2D_RZ topology "
            "(multiblock species remap is not wired; fail-closed refusal)");
      }
      if (burn.neutron_heating) {
        throw ConfigError(
            "Burn.neutron_heating=True is not ported to Main.dimension=\"2D_RZ\" "
            "(v2-E in-flight neutron heating is 1D-only; fail-closed refusal)");
      }
      if (numerics.hydro.hllc_z_flux_2d_rz) {
        throw ConfigError(
            "Burn.enabled=True with Numerics.hydro.hllc_z_flux_2d_rz=True is "
            "not supported (Eulerian z-flux moves mass outside the species "
            "remap; fail-closed refusal)");
      }
    } else {
      throw ConfigError(
          "Burn.enabled=True is supported only for Main.dimension=\"1D_SPH\" "
          "or \"2D_RZ\"");
    }
    for (const auto& fname : burn.fuel_materials) {
      const auto mat_it = std::find_if(materials.materials.begin(),
                                       materials.materials.end(),
                                       [&fname](const auto& mat) {
                                         return mat.name == fname;
                                       });
      if (mat_it == materials.materials.end()) {
        throw ConfigError("Burn.fuel_materials entry \"" + fname +
                          "\" does not match any declared Material name");
      }
      if (mat_it->is_void) {
        throw ConfigError("Burn.fuel_materials entry \"" + fname +
                          "\" must not be a void material");
      }
    }
    const auto contains_fuel = [&burn](const std::string& fuel) {
      return std::find(burn.fuels.begin(), burn.fuels.end(), fuel) !=
             burn.fuels.end();
    };
    const bool has_dd = contains_fuel("DD");
    const bool has_d3he = contains_fuel("D3He");
    // Burn.partition is inert under diffusion and MC transport.
    if ((burn.scheme == "fraley" || burn.scheme == "local") &&
        burn.partition == "fraley" && (has_dd || has_d3he)) {
      throw ConfigError(
          "Burn.partition=\"fraley\" is defined for the DT alpha only (v1); "
          "use partition=\"li_petrasso\" with DD/D3He channels");
    }
  }
  if (laser.enabled && laser.hot_electron.enable) {
    if (laser.hot_electron.tpd_overlap_mode != "single_beam" &&
        laser.hot_electron.tpd_overlap_mode != "common_wave_cluster") {
      throw ConfigError(
          "Laser.hot_electron.tpd_overlap_mode must be \"single_beam\" or "
          "\"common_wave_cluster\"");
    }
    if (laser.hot_electron.tpd_overlap_mode == "common_wave_cluster") {
      if (laser.port_configuration.ports.empty()) {
        throw ConfigError(
            "Laser.hot_electron.tpd_overlap_mode=\"common_wave_cluster\" "
            "requires Laser.port_configuration.ports");
      }
      if (laser.hot_electron.eta_mode != "model") {
        throw ConfigError(
            "Laser.hot_electron.tpd_overlap_mode=\"common_wave_cluster\" "
            "requires Laser.hot_electron.eta_mode=\"model\"");
      }
      if (!(laser.hot_electron.common_wave_delta_theta_deg > 0.0) ||
          laser.hot_electron.common_wave_delta_theta_deg > 90.0) {
        throw ConfigError(
            "Laser.hot_electron.common_wave_delta_theta_deg must be in "
            "(0, 90] for tpd_overlap_mode=\"common_wave_cluster\"");
      }
    }
    if (laser.hot_electron.srs_overlap_mode != "per_beam_class") {
      throw ConfigError(
          "Laser.hot_electron.srs_overlap_mode must be \"per_beam_class\"");
    }
    if (laser.hot_electron.illumination_metric != "fixed" &&
        laser.hot_electron.illumination_metric != "equivalent_area") {
      throw ConfigError(
          "Laser.hot_electron.illumination_metric must be \"fixed\" or "
          "\"equivalent_area\"");
    }
    if (laser.hot_electron.illumination_metric == "equivalent_area") {
      if (laser.port_configuration.ports.empty()) {
        throw ConfigError(
            "Laser.hot_electron.illumination_metric=\"equivalent_area\" "
            "requires Laser.port_configuration.ports");
      }
      if (laser.hot_electron.eta_mode != "model") {
        throw ConfigError(
            "Laser.hot_electron.illumination_metric=\"equivalent_area\" "
            "requires Laser.hot_electron.eta_mode=\"model\"");
      }
    }
    if (laser.hot_electron.eta_mode != "legacy" &&
        laser.hot_electron.eta_mode != "model") {
      throw ConfigError(
          "Laser.hot_electron.eta_mode must be \"legacy\" or \"model\", got \"" +
          laser.hot_electron.eta_mode + "\"");
    }
    if (laser.hot_electron.eta_mode == "model") {
      if (main.dimension != "1D_SPH") {
        throw ConfigError(
            "hot_electron eta_mode=\"model\" is 1D_SPH-only "
            "(2D capture uses segment-start power; blocked by design doc §17.1)");
      }
      if (!laser.hot_electron.sources_specified) {
        throw ConfigError(
            "Laser.hot_electron.eta_mode=\"model\" requires "
            "Laser.hot_electron.sources");
      }
      if (!laser.hot_electron.subtract_from_laser) {
        throw ConfigError(
            "Laser.hot_electron.eta_mode=\"model\" requires "
            "Laser.hot_electron.subtract_from_laser=True");
      }
      if (!(laser.hot_electron.eta_model.eta_total_cap > 0.0) ||
          laser.hot_electron.eta_model.eta_total_cap > 0.5) {
        throw ConfigError(
            "Laser.hot_electron.eta_model.eta_total_cap must be in (0, 0.5]");
      }
      if (!(laser.hot_electron.eta_model.ln_filter_tau_s > 0.0)) {
        throw ConfigError(
            "Laser.hot_electron.eta_model.ln_filter_tau_s must be > 0");
      }
      for (std::size_t si = 0; si < laser.hot_electron.sources.size(); ++si) {
        const auto& channel = laser.hot_electron.sources[si];
        const std::string path =
            "Laser.hot_electron.sources[" + std::to_string(si) + "]";
        if (channel.mechanism != "tpd" && channel.mechanism != "srs") {
          throw ConfigError(
              path + ".mechanism must be \"tpd\" or \"srs\" when "
              "Laser.hot_electron.eta_mode=\"model\"");
        }
        if (channel.eta != 0.0 || channel.eta_table.detected) {
          throw ConfigError(
              path + " must not set eta or eta_table when "
              "Laser.hot_electron.eta_mode=\"model\" (model owns eta)");
        }
        if (!(channel.eval_nc_fraction > 0.0) ||
            !(channel.eval_nc_fraction < 1.0)) {
          throw ConfigError(path + ".eval_nc_fraction must be in (0, 1)");
        }
        if (!(channel.threshold_multiplier > 0.0)) {
          throw ConfigError(path + ".threshold_multiplier must be > 0");
        }
        if (!(channel.eta_inf > 0.0) || !(channel.eta_inf < 1.0)) {
          throw ConfigError(path + ".eta_inf must be in (0, 1)");
        }
        if (!(channel.eta_hard_cap > 0.0) ||
            !(channel.eta_hard_cap < 1.0)) {
          throw ConfigError(path + ".eta_hard_cap must be in (0, 1)");
        }
        if (!(channel.shape_coefficient > 0.0)) {
          throw ConfigError(path + ".shape_coefficient must be > 0");
        }
        if (channel.relaxation_model != "vu2012" &&
            channel.relaxation_model != "fixed") {
          throw ConfigError(
              path + ".relaxation_model must be \"vu2012\" or \"fixed\"");
        }
        if (!(channel.relaxation_tau_s > 0.0)) {
          throw ConfigError(path + ".relaxation_tau_s must be > 0");
        }
        if (!(channel.relaxation_tau_min_s > 0.0) ||
            !(channel.relaxation_tau_min_s <= channel.relaxation_tau_max_s)) {
          throw ConfigError(
              path + ".relaxation_tau_min_s must satisfy "
              "0 < relaxation_tau_min_s <= relaxation_tau_max_s");
        }
      }
    }
    if (main.dimension == "2D_RZ") {
      if (laser.mode != "raytrace_3d") {
        throw ConfigError(
            "Laser.hot_electron.enable=True requires Laser.mode=\"raytrace_3d\" (2D_RZ)");
      }
      if (laser.hot_electron.angular_model == "radial") {
        throw ConfigError(
            "Laser.hot_electron.angular_model=\"radial\" is 1D-only; 2D_RZ requires \"cone\"");
      }
      if (laser.hot_electron.inner_bc != "deposit_residual") {
        throw ConfigError(
            "Laser.hot_electron.inner_bc applies to the 1D radial mode only; "
            "2D_RZ chord transport has no inner endpoint");
      }
    } else if (main.dimension == "1D_SPH") {
      if (laser.mode != "radial_absorption_1d" && laser.mode != "raytrace_2d") {
        throw ConfigError(
            "Laser.hot_electron.enable=True requires Laser.mode=\"radial_absorption_1d\" or "
            "\"raytrace_2d\" (v1)");
      }
      if (laser.hot_electron.angular_model == "cone" &&
          config.mesh.geometry_1d == "cylindrical") {
        throw ConfigError(
            "Laser.hot_electron.angular_model=\"cone\" is not supported for "
            "Mesh.geometry_1d=\"cylindrical\" (v2; use angular_model=\"radial\")");
      }
    } else {
      throw ConfigError(
          "Laser.hot_electron.enable=True is supported only for Main.dimension="
          "\"1D_SPH\" or \"2D_RZ\"");
    }
    if (laser.cbet.enable &&
        !(laser.cbet.geometry_mode == "port_section" &&
          laser.hot_electron.eta_mode == "model")) {
      throw ConfigError(
          "Laser.hot_electron.enable=True is mutually exclusive with Laser.cbet.enable (v1)");
    }
  }
  if (laser.enabled && laser.cbet.enable &&
      !(laser.cbet.ne_frac_cutoff <= 1.0 - laser.absorption.eps_n)) {
    // Inside the active band the CBET gain prefactor and IAW wavenumber floor
    // (1 - n_hat) at eps_n; a cutoff above 1 - eps_n would let the floor
    // silently saturate the gain instead of masking the cell (2026-07-26 review).
    throw ConfigError(
        "Laser.cbet.ne_frac_cutoff must be <= 1 - Laser.absorption.eps_n "
        "(the eps_n floor would saturate the CBET gain inside the active band)");
  }
  if (numerics.persistent_loop.chunk_steps < 1) {
    throw ValueError("Numerics.persistent_loop.chunk_steps must be >= 1");
  }
  if (!(main.t_end > 0.0)) {
    throw ValueError("Main.t_end must be > 0");
  }
  if (mesh.logical_mesh_2d == "cone_shell") {
    constexpr double kPi = 3.141592653589793238462643383279502884;
    if (mesh.topology_scheme_explicit &&
        mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK) {
      throw ConfigError(
          "cone_shell rejects an explicit Mesh.topology_scheme other than "
          "the default \"single_block\"");
    }
    mesh.topology_scheme = TopologyScheme::CONE_SHELL_SPINE;
    if (!mesh.explicit_nodes.empty() || !mesh.grid_segments.empty() ||
        !mesh.auto_regions.empty()) {
      throw ConfigError(
          "cone_shell is exclusive with Mesh.explicit_nodes, Mesh.grid_r, "
          "and Mesh.auto_regions");
    }
    if (!(std::isfinite(mesh.cone_shell_alpha) &&
          mesh.cone_shell_alpha > 0.0 &&
          mesh.cone_shell_alpha < 0.5 * kPi)) {
      throw ValueError(
          "Mesh.cone_shell_alpha must be finite and in (0, pi/2)");
    }
    if (!(std::isfinite(mesh.cone_shell_wall_thickness) &&
          mesh.cone_shell_wall_thickness > 0.0)) {
      throw ValueError(
          "Mesh.cone_shell_wall_thickness must be finite and > 0");
    }
    if (!(std::isfinite(mesh.cone_shell_tip_radius) &&
          mesh.cone_shell_tip_radius > 0.0)) {
      throw ValueError("Mesh.cone_shell_tip_radius must be finite and > 0");
    }
    if (mesh.cone_shell_tip_radius_kind != "inner_face" &&
        mesh.cone_shell_tip_radius_kind != "mid_surface") {
      throw ValueError(
          "Mesh.cone_shell_tip_radius_kind must be one of "
          "{\"inner_face\", \"mid_surface\"}");
    }
    if (!std::isfinite(mesh.cone_shell_tip_z)) {
      throw ValueError("Mesh.cone_shell_tip_z must be finite");
    }
    if (!(std::isfinite(mesh.cone_shell_wall_length) &&
          mesh.cone_shell_wall_length > 0.0)) {
      throw ValueError("Mesh.cone_shell_wall_length must be finite and > 0");
    }
    if (mesh.cone_shell_axis_sign != -1 &&
        mesh.cone_shell_axis_sign != 1) {
      throw ValueError("Mesh.cone_shell_axis_sign must be -1 or +1");
    }
    if (mesh.cone_shell_n_cells < 6 ||
        mesh.cone_shell_n_cells % 2 != 0) {
      throw ValueError(
          "Mesh.cone_shell_n_cells must be even and >= 6");
    }
    if (!(std::isfinite(mesh.cone_shell_n_growth) &&
          mesh.cone_shell_n_growth > 1.0 &&
          mesh.cone_shell_n_growth <= 2.0)) {
      throw ValueError(
          "Mesh.cone_shell_n_growth must be finite and in (1, 2]");
    }
    if (!(std::isfinite(mesh.cone_shell_tip_size_factor) &&
          mesh.cone_shell_tip_size_factor >= 1.0 &&
          mesh.cone_shell_tip_size_factor <= 64.0)) {
      throw ValueError(
          "Mesh.cone_shell_tip_size_factor must be finite and in [1, 64]");
    }
    if (!(std::isfinite(mesh.cone_shell_base_size_factor) &&
          mesh.cone_shell_base_size_factor >=
              mesh.cone_shell_tip_size_factor &&
          mesh.cone_shell_base_size_factor <= 64.0)) {
      throw ValueError(
          "Mesh.cone_shell_base_size_factor must be finite, >= "
          "Mesh.cone_shell_tip_size_factor, and <= 64");
    }
    if (std::isnan(mesh.cone_shell_tip_hold)) {
      mesh.cone_shell_tip_hold = 4.0 * mesh.cone_shell_wall_thickness;
    }
    if (std::isnan(mesh.cone_shell_grading_length)) {
      mesh.cone_shell_grading_length =
          std::min(0.35 * mesh.cone_shell_wall_length,
                   20.0 * mesh.cone_shell_wall_thickness);
    }
    if (std::isnan(mesh.cone_shell_tip_rotation_length)) {
      mesh.cone_shell_tip_rotation_length =
          3.0 * mesh.cone_shell_wall_thickness;
    }
    if (std::isnan(mesh.cone_shell_base_rotation_length)) {
      mesh.cone_shell_base_rotation_length =
          mesh.cone_shell_tip_rotation_length;
    }
    if (!(std::isfinite(mesh.cone_shell_tip_hold) &&
          mesh.cone_shell_tip_hold >= 0.0)) {
      throw ValueError(
          "Mesh.cone_shell_tip_hold must be finite and >= 0");
    }
    if (!(std::isfinite(mesh.cone_shell_grading_length) &&
          mesh.cone_shell_grading_length > 0.0)) {
      throw ValueError(
          "Mesh.cone_shell_grading_length must be finite and > 0");
    }
    if (!(std::isfinite(mesh.cone_shell_l_ratio_max) &&
          mesh.cone_shell_l_ratio_max > 1.0 &&
          mesh.cone_shell_l_ratio_max <= 1.5)) {
      throw ValueError(
          "Mesh.cone_shell_l_ratio_max must be finite and in (1, 1.5]");
    }
    if (!(std::isfinite(mesh.cone_shell_tip_rotation_length) &&
          mesh.cone_shell_tip_rotation_length > 0.0)) {
      throw ValueError(
          "Mesh.cone_shell_tip_rotation_length must be finite and > 0");
    }
    if (mesh.cone_shell_base_cut != "planar" &&
        mesh.cone_shell_base_cut != "wall_normal") {
      throw ValueError(
          "Mesh.cone_shell_base_cut must be one of "
          "{\"planar\", \"wall_normal\"}");
    }
    if (!(std::isfinite(mesh.cone_shell_base_rotation_length) &&
          mesh.cone_shell_base_rotation_length > 0.0)) {
      throw ValueError(
          "Mesh.cone_shell_base_rotation_length must be finite and > 0");
    }
    if (mesh.cone_shell_farfield_target_measure != "station_uniform" &&
        mesh.cone_shell_farfield_target_measure != "wall_phi") {
      throw ConfigError(
          "cone_shell_farfield_target_measure must be \"station_uniform\" or "
          "\"wall_phi\"");
    }
    if (mesh.cone_shell_base_cut == "planar" &&
        mesh.cone_shell_tip_rotation_length +
                mesh.cone_shell_base_rotation_length >
            mesh.cone_shell_wall_length) {
      throw ValueError(
          "cone_shell planar tip/base rotation lengths must satisfy "
          "L_t + L_b <= L_w");
    }
    if (!(std::isfinite(mesh.cone_shell_outer_vac_first_factor) &&
          mesh.cone_shell_outer_vac_first_factor > 0.0 &&
          mesh.cone_shell_outer_vac_first_factor <= 4.0)) {
      throw ValueError(
          "Mesh.cone_shell_outer_vac_first_factor must be finite and in (0, 4]");
    }
    if (mesh.cone_shell_outer_vac_layers < 2 ||
        mesh.cone_shell_outer_vac_layers > 64) {
      throw ValueError(
          "Mesh.cone_shell_outer_vac_layers must be in [2, 64]");
    }
    if (!(std::isfinite(mesh.cone_shell_outer_vac_growth) &&
          mesh.cone_shell_outer_vac_growth > 1.0 &&
          mesh.cone_shell_outer_vac_growth <= 2.0)) {
      throw ValueError(
          "Mesh.cone_shell_outer_vac_growth must be finite and in (1, 2]");
    }
    if (!(std::isfinite(mesh.cone_shell_inner_vac_first_factor) &&
          mesh.cone_shell_inner_vac_first_factor > 0.0 &&
          mesh.cone_shell_inner_vac_first_factor <= 4.0)) {
      throw ValueError(
          "Mesh.cone_shell_inner_vac_first_factor must be finite and in (0, 4]");
    }
    if (mesh.cone_shell_inner_vac_layers < 2 ||
        mesh.cone_shell_inner_vac_layers > 64) {
      throw ValueError(
          "Mesh.cone_shell_inner_vac_layers must be in [2, 64]");
    }
    if (!(std::isfinite(mesh.cone_shell_inner_vac_growth) &&
          mesh.cone_shell_inner_vac_growth > 1.0 &&
          mesh.cone_shell_inner_vac_growth <= 2.0)) {
      throw ValueError(
          "Mesh.cone_shell_inner_vac_growth must be finite and in (1, 2]");
    }
    if (!(std::isfinite(mesh.cone_shell_end_vac_first_factor) &&
          mesh.cone_shell_end_vac_first_factor > 0.0 &&
          mesh.cone_shell_end_vac_first_factor <= 4.0)) {
      throw ValueError(
          "Mesh.cone_shell_end_vac_first_factor must be finite and in (0, 4]");
    }
    if (mesh.cone_shell_end_vac_layers < 2 ||
        mesh.cone_shell_end_vac_layers > 64) {
      throw ValueError(
          "Mesh.cone_shell_end_vac_layers must be in [2, 64]");
    }
    if (!(std::isfinite(mesh.cone_shell_end_vac_growth) &&
          mesh.cone_shell_end_vac_growth > 1.0 &&
          mesh.cone_shell_end_vac_growth <= 2.0)) {
      throw ValueError(
          "Mesh.cone_shell_end_vac_growth must be finite and in (1, 2]");
    }
    if (mesh.cone_shell_outer_vac_layers !=
            mesh.cone_shell_inner_vac_layers ||
        mesh.cone_shell_inner_vac_layers !=
            mesh.cone_shell_end_vac_layers) {
      throw ConfigError(
          "cone_shell corner-closure requirement needs "
          "Mesh.cone_shell_outer_vac_layers == "
          "Mesh.cone_shell_inner_vac_layers == "
          "Mesh.cone_shell_end_vac_layers");
    }

    const double cos_alpha = std::cos(mesh.cone_shell_alpha);
    constexpr double kCornerNormalDeterminantFloor = 0.2;
    if (std::abs(cos_alpha) < kCornerNormalDeterminantFloor) {
      constexpr double kRadiansToDegrees =
          57.295779513082320876798154814105170332405472466564;
      const double included_angle_degrees =
          std::asin(std::clamp(std::abs(cos_alpha), 0.0, 1.0)) *
          kRadiansToDegrees;
      std::ostringstream oss;
      oss << "cone_shell corner normal matrix conditioning requires |det| >= "
          << kCornerNormalDeterminantFloor
          << "; included_angle_degrees="
          << std::setprecision(std::numeric_limits<double>::max_digits10)
          << included_angle_degrees;
      throw ConfigError(oss.str());
    }
    const double half_thickness = 0.5 * mesh.cone_shell_wall_thickness;
    const double mid_tip_radius =
        mesh.cone_shell_tip_radius_kind == "inner_face"
            ? mesh.cone_shell_tip_radius + half_thickness / cos_alpha
            : mesh.cone_shell_tip_radius;
    const double inner_tip_radius =
        mid_tip_radius - half_thickness / cos_alpha;
    if (!(inner_tip_radius > 0.0)) {
      throw ValueError(
          "cone_shell tip radius is nonpositive after conversion to the "
          "inner-face truncation radius");
    }

    double rotation_derivative_bound =
        1.875 / mesh.cone_shell_tip_rotation_length;
    if (mesh.cone_shell_base_cut == "planar") {
      rotation_derivative_bound = std::max(
          rotation_derivative_bound,
          1.875 / mesh.cone_shell_base_rotation_length);
    }
    const double analytic_margin =
        1.0 - half_thickness * std::tan(mesh.cone_shell_alpha) *
                  rotation_derivative_bound;
    if (!(analytic_margin >= 0.25)) {
      std::ostringstream oss;
      oss << "cone_shell analytic wall-Jacobian margin must be >= 0.25; got "
          << std::setprecision(std::numeric_limits<double>::max_digits10)
          << analytic_margin;
      throw ConfigError(oss.str());
    }

    const double sigma = static_cast<double>(mesh.cone_shell_axis_sign);
    const double z_base =
        mesh.cone_shell_tip_z +
        sigma * mesh.cone_shell_wall_length * cos_alpha;
    if (!(std::isfinite(mesh.box_r_max) && mesh.box_r_max > 0.0 &&
          std::isfinite(mesh.box_z_min) &&
          std::isfinite(mesh.box_z_max))) {
      throw ValueError(
          "cone_shell C4 requires finite Mesh.box_r_max > 0, "
          "Mesh.box_z_min, and Mesh.box_z_max");
    }
    const double required_box_z = mesh.cone_shell_axis_sign > 0
                                      ? mesh.box_z_max
                                      : mesh.box_z_min;
    if (required_box_z != z_base) {
      std::ostringstream oss;
      oss << "cone_shell base-side box z face must equal the required value "
          << std::setprecision(std::numeric_limits<double>::max_digits10)
          << z_base << " exactly; got " << required_box_z;
      throw ConfigError(oss.str());
    }

    const double h_n0 = tenryu::core::cone_shell_normal_face_width(
        mesh.cone_shell_wall_thickness, mesh.cone_shell_n_cells,
        mesh.cone_shell_n_growth);
    const double end_first =
        mesh.cone_shell_end_vac_first_factor * h_n0;
    const double end_depth = cone_shell_strip_total(
        end_first, mesh.cone_shell_end_vac_layers,
        mesh.cone_shell_end_vac_growth);
    const double tip_room = end_depth + 8.0 * mesh.cone_shell_wall_thickness;
    if (mesh.cone_shell_axis_sign > 0) {
      if (!(mesh.box_z_min < mesh.cone_shell_tip_z - tip_room)) {
        throw ValueError(
            "cone_shell requires Mesh.box_z_min < cone_shell_tip_z - "
            "(end-strip depth + 8*t_w)");
      }
    } else if (!(mesh.box_z_max > mesh.cone_shell_tip_z + tip_room)) {
      throw ValueError(
          "cone_shell requires Mesh.box_z_max > cone_shell_tip_z + "
          "(end-strip depth + 8*t_w)");
    }

    tenryu::core::ConeShellAlongWallSpec ladder_spec{
        mesh.cone_shell_wall_length,
        mesh.cone_shell_wall_thickness,
        mesh.cone_shell_n_cells,
        mesh.cone_shell_n_growth,
        mesh.cone_shell_tip_size_factor,
        mesh.cone_shell_base_size_factor,
        mesh.cone_shell_tip_hold,
        mesh.cone_shell_grading_length,
        mesh.cone_shell_l_ratio_max,
        mesh.cone_shell_tip_rotation_length,
        mesh.cone_shell_base_rotation_length,
        mesh.cone_shell_base_cut == "planar",
    };
    tenryu::core::ConeShellAlongWallLadder ladder =
        tenryu::core::build_cone_shell_along_wall_ladder(ladder_spec);
    if (!ladder.counts_valid || ladder.q.size() < 2U ||
        ladder.q.size() - 1U >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw ConfigError("cone_shell along-wall cell count is invalid");
    }

    const double outer_first =
        mesh.cone_shell_outer_vac_first_factor * h_n0;
    const double outer_depth = cone_shell_strip_total(
        outer_first, mesh.cone_shell_outer_vac_layers,
        mesh.cone_shell_outer_vac_growth);
    tenryu::core::ConeShellExteriorRaySpec ray_spec{
        mesh.cone_shell_wall_length,
        mesh.cone_shell_wall_thickness,
        mesh.cone_shell_alpha,
        mid_tip_radius,
        mesh.cone_shell_tip_z,
        mesh.cone_shell_axis_sign,
        mesh.cone_shell_tip_rotation_length,
        mesh.cone_shell_base_rotation_length,
        mesh.cone_shell_base_cut == "planar",
        outer_depth,
        mesh.box_r_max,
        mesh.box_z_min,
        mesh.box_z_max,
    };
    double outer_far_r_max = 0.0;
    for (const double q : ladder.q) {
      outer_far_r_max = std::max(
          outer_far_r_max,
          tenryu::core::cone_shell_outer_far_point(q, ray_spec).r);
    }
    const double required_box_r_min =
        outer_far_r_max + 2.0 * mesh.cone_shell_wall_thickness;
    if (!(mesh.box_r_max > required_box_r_min)) {
      std::ostringstream oss;
      oss << "cone_shell Mesh.box_r_max must exceed "
          << std::setprecision(std::numeric_limits<double>::max_digits10)
          << required_box_r_min
          << " (maximum outer-strip far-edge station r + 2*t_w)";
      throw ConfigError(oss.str());
    }

    mesh.nr = static_cast<int>(ladder.q.size()) - 1;
    mesh.nz = mesh.cone_shell_n_cells;

    const double inner_first =
        mesh.cone_shell_inner_vac_first_factor * h_n0;
    const double inner_depth = cone_shell_strip_total(
        inner_first, mesh.cone_shell_inner_vac_layers,
        mesh.cone_shell_inner_vac_growth);
    const double inner_h_start = cone_shell_strip_last_width(
        inner_first, mesh.cone_shell_inner_vac_layers,
        mesh.cone_shell_inner_vac_growth);
    const double outer_h_start = cone_shell_strip_last_width(
        outer_first, mesh.cone_shell_outer_vac_layers,
        mesh.cone_shell_outer_vac_growth);
    double cavity_distance_min = std::numeric_limits<double>::infinity();
    double cavity_distance_max = 0.0;
    double exterior_distance_min = std::numeric_limits<double>::infinity();
    double exterior_distance_max = 0.0;
    const double exterior_start_z =
        tenryu::core::cone_shell_outer_far_point(0.0, ray_spec).z;
    for (const double q : ladder.q) {
      const tenryu::core::ConeShellPoint direction =
          tenryu::core::cone_shell_through_direction(q, ray_spec);
      const tenryu::core::ConeShellPoint inner_wall =
          tenryu::core::cone_shell_wall_point(q, -half_thickness, ray_spec);
      const tenryu::core::ConeShellPoint inner_far{
          inner_wall.r - inner_depth * direction.r,
          inner_wall.z - inner_depth * direction.z,
      };
      const double phi =
          tenryu::core::cone_shell_along_wall_monitor_measure(
              q, ladder_spec);
      const double axis_z =
          mesh.cone_shell_tip_z + phi * (z_base - mesh.cone_shell_tip_z);
      // The Hermite bridge uses the chord as its curve-arclength
      // approximation for the common mixed-ladder rule.
      const double cavity_distance =
          std::hypot(inner_far.r, inner_far.z - axis_z);
      if (!(std::isfinite(cavity_distance) && cavity_distance > 0.0)) {
        throw ConfigError(
            "cone_shell cavity Hermite chord must be positive");
      }
      cavity_distance_min =
          std::min(cavity_distance_min, cavity_distance);
      cavity_distance_max =
          std::max(cavity_distance_max, cavity_distance);

      const tenryu::core::ConeShellPoint outer_far =
          tenryu::core::cone_shell_outer_far_point(q, ray_spec);
      double exterior_target_z =
          exterior_start_z + phi * (z_base - exterior_start_z);
      if (q == 0.0) {
        exterior_target_z = outer_far.z;
      } else if (q == mesh.cone_shell_wall_length) {
        exterior_target_z = z_base;
      }
      // The exterior Hermite bridge uses the chord as its curve-arclength
      // approximation for the common mixed-ladder rule.
      const double exterior_distance = std::hypot(
          mesh.box_r_max - outer_far.r,
          exterior_target_z - outer_far.z);
      if (!(std::isfinite(exterior_distance) &&
            exterior_distance > 0.0)) {
        throw ConfigError(
            "cone_shell exterior Hermite chord must be positive");
      }
      exterior_distance_min =
          std::min(exterior_distance_min, exterior_distance);
      exterior_distance_max =
          std::max(exterior_distance_max, exterior_distance);
    }
    mesh.cone_shell_cavity_cells = cone_shell_mixed_ray_count(
        inner_h_start, cavity_distance_min, cavity_distance_max,
        "CAVITY_CORE");
    mesh.cone_shell_exterior_cells = cone_shell_mixed_ray_count(
        outer_h_start, exterior_distance_min, exterior_distance_max,
        "EXTERIOR_CORE");
    mesh.cone_shell_tip_fill_layers =
        cone_shell_tip_fill_layer_count(mesh, h_n0);

    mesh.r_min = 0.0;
    mesh.r_max = mesh.box_r_max;
    mesh.z_min = mesh.box_z_min;
    mesh.z_max = mesh.box_z_max;
  } else if (mesh.topology_scheme == TopologyScheme::CONE_SHELL_SPINE) {
    throw ConfigError(
        "Mesh.topology_scheme=\"cone_shell_spine\" is builder-derived and "
        "cannot be supplied by a deck");
  }
  const bool uses_physical_r_bounds =
      is_dimension_1d(main.dimension) || mesh.logical_mesh_2d == "rectangular_rz";
  if (uses_physical_r_bounds && !std::isnan(mesh.r_min) && mesh.r_min < 0.0) {
    throw ConfigError("Mesh.r_min must be >= 0");
  }
  if (uses_physical_r_bounds &&
      (std::isnan(mesh.r_min) || std::isnan(mesh.r_max) || !(mesh.r_max > mesh.r_min))) {
    throw ConfigError("Mesh requires r_min and r_max with r_max > r_min");
  }
  if (is_dimension_1d(main.dimension)) {
    mesh.grid_type_r = "graded";
  }
  if (mesh.grid_type_r == "graded" && !mesh.grid_segments.empty()) {
    validate_mesh_grading(mesh.grading);
  }
  if (mesh.grid_type_z == "graded" && !mesh.grid_segments_z.empty()) {
    validate_mesh_grading(mesh.grading);
  }
  if (!(std::isfinite(mesh.polar_theta_min) &&
        mesh.polar_theta_min >= 0.0 && mesh.polar_theta_min < 2.6)) {
    throw ValueError("Mesh.polar_theta_min must be finite and in [0, 2.6)");
  }
  if (mesh.polar_theta_min > 0.0) {
    if (mesh.logical_mesh_2d != "spherical_polar_halfplane") {
      throw ConfigError(
          "Mesh.polar_theta_min > 0 requires "
          "Mesh.logical_mesh_2d='spherical_polar_halfplane'");
    }
    if (mesh.polar_center_treatment != "tri_fan") {
      throw ConfigError(
          "Mesh.polar_theta_min > 0 requires the tri_fan center (v1)");
    }
  }
  if (main.dimension == "2D_RZ") {
    if (mesh.grid_type_z == "graded") {
      if (tenryu::core::is_polar_family(mesh.logical_mesh_2d)) {
        throw ConfigError(
            "graded z is not applicable to polar-family meshes (the z index "
            "is angular)");
      }
      if (mesh.grid_segments_z.empty() && mesh.explicit_nodes_z.empty()) {
        throw ConfigError(
            "2D_RZ Mesh.grid_type_z='graded' requires Mesh.grid_z segments");
      }
    } else if (mesh.grid_type_z != "uniform") {
      throw ConfigError("Mesh.grid_type_z must be 'uniform' or 'graded'");
    }
    const bool has_theta_zoning = !mesh.grid_segments_theta.empty() ||
                                  !mesh.explicit_nodes_theta.empty();
    if (has_theta_zoning) {
      if (!tenryu::core::is_polar_family(mesh.logical_mesh_2d)) {
        throw ConfigError(
            "Mesh.grid_theta/explicit_nodes_theta apply only to "
            "polar-family logical_mesh_2d values");
      }
      if (mesh.polar_equal_mu_zoning) {
        throw ConfigError(
            "Mesh.grid_theta/explicit_nodes_theta are exclusive with "
            "Mesh.polar_equal_mu_zoning");
      }
      if (!mesh.grid_segments_theta.empty() &&
          !mesh.explicit_nodes_theta.empty()) {
        throw ConfigError(
            "Mesh.grid_theta and Mesh.explicit_nodes_theta are exclusive");
      }

      constexpr double kPiTheta = 3.14159265358979323846;
      int n_theta_total = 0;
      if (!mesh.grid_segments_theta.empty()) {
        if (mesh.grid_segments_theta.front().r_start !=
            mesh.polar_theta_min) {
          throw ConfigError(
              "Mesh.grid_theta segments must start at "
              "Mesh.polar_theta_min");
        }
        if (!(std::abs(mesh.grid_segments_theta.back().r_end - kPiTheta) <=
              1.0e-12)) {
          throw ConfigError(
              "Mesh.grid_theta segments must end at theta = pi");
        }
        for (const auto& segment : mesh.grid_segments_theta) {
          n_theta_total += segment.nr;
        }
      } else {
        if (mesh.explicit_nodes_theta.size() < 2) {
          throw ConfigError(
              "Mesh.explicit_nodes_theta must contain at least 2 nodes");
        }
        if (mesh.explicit_nodes_theta.front() != mesh.polar_theta_min) {
          throw ConfigError(
              "Mesh.explicit_nodes_theta must start at "
              "Mesh.polar_theta_min");
        }
        if (!(std::abs(mesh.explicit_nodes_theta.back() - kPiTheta) <=
              1.0e-12)) {
          throw ConfigError(
              "Mesh.explicit_nodes_theta must end at theta = pi");
        }
        for (std::size_t k = 1; k < mesh.explicit_nodes_theta.size(); ++k) {
          if (!(mesh.explicit_nodes_theta[k] >
                mesh.explicit_nodes_theta[k - 1])) {
            throw ConfigError(
                "Mesh.explicit_nodes_theta must be strictly increasing");
          }
        }
        n_theta_total =
            static_cast<int>(mesh.explicit_nodes_theta.size()) - 1;
      }
      if (mesh.nz != 1 && mesh.nz != n_theta_total) {
        throw ConfigError(
            "Mesh.nz (" + std::to_string(mesh.nz) +
            ") conflicts with Mesh.grid_theta/explicit_nodes_theta total "
            "zones (" + std::to_string(n_theta_total) +
            "); omit Mesh.nz (default nz=1 is treated as unset) or match it");
      }
      mesh.nz = n_theta_total;
    }
  }
  if (is_dimension_1d(main.dimension)) {
    if (!mesh.grid_segments_z.empty() || !mesh.explicit_nodes_z.empty()) {
      throw ConfigError(
          "Mesh.grid_z applies only to Main.dimension='2D_RZ'");
    }
    if (!mesh.grid_segments_theta.empty() ||
        !mesh.explicit_nodes_theta.empty()) {
      throw ConfigError(
          "Mesh.grid_theta applies only to 2D_RZ "
          "spherical_polar_halfplane meshes");
    }
  }
  if (mesh.logical_mesh_2d == "polar_in_box") {
    if (mesh.explicit_nodes.empty()) {
      throw ConfigError(
          "polar_in_box requires Mesh.explicit_nodes for the polar prefix ladder");
    }
    mesh.polar_prefix_nr =
        static_cast<int>(mesh.explicit_nodes.size()) - 1;
    mesh.nr =
        mesh.polar_prefix_nr + mesh.morph_rings + mesh.collar_rings;
    if (mesh.topology_scheme == TopologyScheme::SINGLE_BLOCK &&
        mesh.polar_center_treatment != "annular" &&
        mesh.polar_center_treatment != "tri_fan") {
      throw ConfigError(
          "polar_in_box general-quad initialization requires "
          "Mesh.polar_center_treatment='annular'");
    }
    if (mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK &&
        mesh.topology_scheme !=
            TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL) {
      throw ConfigError(
          "polar_in_box supports single_block or "
          "multiblock_cart_core_polar_shell topology");
    }
  }
  if (mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK) {
    if (!mesh.explicit_nodes.empty()) {
      if (mesh.topology_scheme ==
          TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL) {
        if (mesh.logical_mesh_2d == "polar_in_box" &&
            !(mesh.multiblock_cart_core_r_match <
              mesh.explicit_nodes.back())) {
          throw ConfigError(
              "polar_in_box + multiblock_cart_core_polar_shell requires "
              "multiblock_cart_core_r_match < the polar lock radius "
              "(Mesh.explicit_nodes.back())");
        }
        if (mesh.explicit_nodes.front() !=
            mesh.multiblock_cart_core_r_match) {
          std::ostringstream oss;
          oss << "Mesh.explicit_nodes with cart-core topology must start "
                 "exactly at multiblock_cart_core_r_match (got front="
              << std::setprecision(std::numeric_limits<double>::max_digits10)
              << mesh.explicit_nodes.front()
              << " vs r_match=" << mesh.multiblock_cart_core_r_match << ")";
          throw ConfigError(oss.str());
        }
        if (mesh.logical_mesh_2d != "polar_in_box" &&
            mesh.explicit_nodes.back() != mesh.spherical_polar_s_max) {
          std::ostringstream oss;
          oss << "Mesh.explicit_nodes with cart-core topology must end "
                 "exactly at spherical_polar_s_max (got back="
              << std::setprecision(std::numeric_limits<double>::max_digits10)
              << mesh.explicit_nodes.back()
              << " vs s_max=" << mesh.spherical_polar_s_max << ")";
          throw ConfigError(oss.str());
        }
      } else {
        throw ConfigError(
            "Mesh.explicit_nodes is not supported with multiblock "
            "topology_scheme (use grid segments for the shell)");
      }
    }
    if (!mesh.auto_regions.empty()) {
      throw ConfigError(
          "Mesh.auto_regions is not supported with multiblock "
          "topology_scheme yet (use grid segments for the shell)");
    }
  }
  if (!mesh.auto_regions.empty()) {
    const bool is_2d = (main.dimension == "2D_RZ");
    const bool is_polar =
        is_2d && mesh.logical_mesh_2d == "spherical_polar_halfplane";
    const bool auto_axis_z = (mesh.auto_regions_axis == "z");
    if (auto_axis_z &&
        (!is_2d || mesh.logical_mesh_2d != "rectangular_rz")) {
      throw ConfigError(
          "Mesh.auto_regions_axis='z' requires 2D_RZ rectangular_rz");
    }
    if (auto_axis_z) {
      if (!mesh.grid_segments_z.empty()) {
        throw ConfigError(
            "Mesh.auto_regions and Mesh.grid_z segments are mutually exclusive");
      }
    } else if (!mesh.grid_segments.empty()) {
      throw ConfigError(
          "Mesh.auto_regions and Mesh.grid segments are mutually exclusive");
    }
    if (is_polar && mesh.polar_center_treatment == "annular") {
      throw ConfigError(
          "Mesh.auto_regions with polar_center_treatment='annular' is not "
          "supported; use Mesh.grid segments instead");
    }
    tenryu::core::AutoZoneConfig az_cfg;
    az_cfg.mass_ratio_max = mesh.auto_config.mass_ratio_max;
    az_cfg.n_bridge_min = mesh.auto_config.n_bridge_min;
    az_cfg.n_bridge_max = mesh.auto_config.n_bridge_max;
    az_cfg.bridge_frac_max = mesh.auto_config.bridge_frac_max;
    az_cfg.rho_void_cut = mesh.auto_config.rho_void_cut;
    az_cfg.dr_min = mesh.auto_config.dr_min;
    az_cfg.mass_ratio_hard_max = mesh.auto_config.mass_ratio_hard_max;
    az_cfg.max_iter = mesh.auto_config.max_iter;
    az_cfg.bulk_mass_tol = mesh.auto_config.bulk_mass_tol;
    if (auto_axis_z) {
      az_cfg.geometry_code = 2;
    } else if (is_polar) {
      az_cfg.geometry_code = 0;
    } else if (is_2d) {
      az_cfg.geometry_code = 1;
    } else {
      az_cfg.geometry_code =
          (mesh.geometry_1d == "cylindrical")
              ? 1
              : ((mesh.geometry_1d == "planar") ? 2 : 0);
    }
    double r_inner = 0.0;
    double r_outer_expected = 0.0;
    if (auto_axis_z) {
      if (!std::isfinite(mesh.z_min) || !std::isfinite(mesh.z_max)) {
        throw ConfigError("Mesh.auto_regions_axis='z' requires z_min and z_max");
      }
      r_inner = mesh.z_min;
      r_outer_expected = mesh.z_max;
    } else if (is_polar) {
      r_inner = 0.0;
      r_outer_expected = mesh.spherical_polar_s_max;
    } else {
      if (std::isnan(mesh.r_min) || std::isnan(mesh.r_max)) {
        throw ConfigError("Mesh.auto_regions requires r_min and r_max");
      }
      r_inner = mesh.r_min;
      r_outer_expected = mesh.r_max;
    }
    const double r_last = mesh.auto_regions.back().r_end;
    if (std::abs(r_last - r_outer_expected) >
        1.0e-12 * std::max(1.0, std::abs(r_outer_expected))) {
      throw ConfigError(
          "Mesh.auto_regions last r_end must equal the outer boundary (" +
          std::to_string(r_outer_expected) + "), got " +
          std::to_string(r_last));
    }
    std::vector<tenryu::core::AutoZoneRegion> az_regions;
    az_regions.reserve(mesh.auto_regions.size());
    for (const auto& region : mesh.auto_regions) {
      tenryu::core::AutoZoneRegion az;
      az.r_end = region.r_end;
      az.nz = region.nz;
      az.rho_ref = region.rho_ref;
      az.is_void = region.is_void;
      az.material_group = region.material_group;
      az_regions.push_back(az);
    }
    tenryu::core::AutoZoneDiagnostics az_diag;
    std::vector<double>& explicit_nodes =
        auto_axis_z ? mesh.explicit_nodes_z : mesh.explicit_nodes;
    explicit_nodes =
        tenryu::core::compute_auto_zone_nodes(r_inner, az_regions, az_cfg,
                                              &az_diag);
    const int n_auto = static_cast<int>(explicit_nodes.size()) - 1;
    if (auto_axis_z) {
      if (mesh.nz != 1 && mesh.nz != n_auto) {
        throw ConfigError(
            "Mesh.nz (" + std::to_string(mesh.nz) +
            ") conflicts with Mesh.auto_regions total zones (" +
            std::to_string(n_auto) +
            "); omit Mesh.nz (default nz=1 is treated as unset) or match it");
      }
      mesh.nz = n_auto;
      mesh.grid_type_z = "graded";
    } else {
      if (mesh.nr >= 0 && mesh.nr != n_auto) {
        throw ConfigError(
            "Mesh.nr (" + std::to_string(mesh.nr) +
            ") conflicts with Mesh.auto_regions total zones (" +
            std::to_string(n_auto) + "); omit Mesh.nr or match it");
      }
      mesh.nr = n_auto;
    }
    std::string az_log =
        "[mesh-autozone] nodes=" + std::to_string(explicit_nodes.size()) +
        " axis=" + mesh.auto_regions_axis +
        " mass_ratio[min/mean/max]=" + std::to_string(az_diag.mass_ratio_min) +
        "/" + std::to_string(az_diag.mass_ratio_mean) + "/" +
        std::to_string(az_diag.mass_ratio_max) +
        " violations=" + std::to_string(az_diag.n_ratio_violations);
    tenryu::core::log_info(az_log);
    for (const auto& w : az_diag.warnings) {
      tenryu::core::log_warning("[mesh-autozone] " + w);
    }
  }
  if (mesh.zoning_intent.enabled) {
    const auto& zoning = mesh.zoning_intent;
    if (!mesh.auto_regions.empty()) {
      throw ConfigError(
          "Mesh.zoning_intent and Mesh.auto_regions are mutually exclusive");
    }
    if (!mesh.grid_segments.empty()) {
      throw ConfigError(
          "Mesh.zoning_intent and Mesh.grid segments are mutually exclusive");
    }
    if (main.dimension == "2D_RZ") {
      throw ConfigError(
          "Mesh.zoning_intent supports 1D runs only in this version");
    }
    std::string required_geometry;
    if (zoning.measure == "spherical_cell_mass") {
      required_geometry = "spherical";
    } else if (zoning.measure == "cylindrical_line_mass") {
      required_geometry = "cylindrical";
    }
    if (!required_geometry.empty() &&
        mesh.geometry_1d != required_geometry) {
      throw ConfigError(
          "Mesh.zoning_intent.measure='" + zoning.measure +
          "' requires Geometry '" + required_geometry + "', got '" +
          mesh.geometry_1d +
          "'; 'width' and 'areal_mass' are geometry-independent");
    }
    if (!std::isfinite(mesh.r_min) || !std::isfinite(mesh.r_max)) {
      throw ConfigError("Mesh.zoning_intent requires r_min and r_max");
    }

    const bool is_width = (zoning.measure == "width");
    if (is_width) {
      if (!zoning.density_regions.empty()) {
        tenryu::core::log_warning(
            "[mesh-zoning-intent] density_regions ignored for "
            "measure='width'");
      }
    } else {
      if (zoning.density_regions.empty()) {
        throw ConfigError("Mesh.zoning_intent." + zoning.measure +
                          " requires density_regions");
      }
      const double r_last = zoning.density_regions.back().r_end;
      if (std::abs(r_last - mesh.r_max) >
          1.0e-12 * std::max(1.0, std::abs(mesh.r_max))) {
        throw ConfigError(
            "Mesh.zoning_intent.density_regions last r_end must equal the "
            "outer boundary (" +
            std::to_string(mesh.r_max) + "), got " +
            std::to_string(r_last));
      }
    }

    tenryu::core::ZoningIntentConfig zic;
    zic.n_cells = zoning.n_cells;
    zic.dr_min = zoning.dr_min;
    zic.cell_measure_min = zoning.cell_measure_min;
    zic.cell_measure_max = zoning.cell_measure_max;
    zic.preferred_ratio = zoning.preferred_ratio;
    zic.ratio_hard_max = zoning.ratio_hard_max;
    zic.min_cells_per_segment = zoning.min_cells_per_segment;
    if (zoning.measure == "width") {
      zic.measure = tenryu::core::ZoningMeasure::kWidth;
    } else if (zoning.measure == "areal_mass") {
      zic.measure = tenryu::core::ZoningMeasure::kArealMass;
    } else if (zoning.measure == "cylindrical_line_mass") {
      zic.measure = tenryu::core::ZoningMeasure::kCylindricalLineMass;
    } else {
      zic.measure = tenryu::core::ZoningMeasure::kSphericalCellMass;
    }
    zic.pins.reserve(zoning.pins.size());
    for (const auto& pin : zoning.pins) {
      tenryu::core::ZoningIntentPin out;
      out.r = pin.r;
      out.ratio_jump_allowed = pin.ratio_jump_allowed;
      zic.pins.push_back(out);
    }
    zic.profile.reserve(zoning.profile.size());
    for (const auto& point : zoning.profile) {
      tenryu::core::ZoningProfilePoint out;
      out.r = point.r;
      out.w = point.w;
      zic.profile.push_back(out);
    }
    zic.anchors.reserve(zoning.anchors.size());
    for (const auto& anchor : zoning.anchors) {
      tenryu::core::ZoningAnchor out;
      out.r = anchor.r;
      out.half_width = anchor.half_width;
      out.log_amplitude = anchor.log_amplitude;
      zic.anchors.push_back(out);
    }
    zic.bands.reserve(zoning.bands.size());
    for (const auto& band : zoning.bands) {
      tenryu::core::ZoningBand out;
      out.measure_frac_begin = band.measure_frac_begin;
      out.measure_frac_end = band.measure_frac_end;
      out.cell_measure_min = band.cell_measure_min;
      out.cell_measure_max = band.cell_measure_max;
      zic.bands.push_back(out);
    }
    zic.extra_events = zoning.extra_events;

    std::function<double(double)> rho0;
    if (!is_width) {
      std::vector<double> r_ends;
      std::vector<double> densities;
      r_ends.reserve(zoning.density_regions.size());
      densities.reserve(zoning.density_regions.size());
      for (const auto& region : zoning.density_regions) {
        r_ends.push_back(region.r_end);
        densities.push_back(region.rho);
        if (region.r_end > mesh.r_min && region.r_end < mesh.r_max) {
          zic.extra_events.push_back(region.r_end);
        }
      }
      rho0 = [r_ends = std::move(r_ends),
              densities = std::move(densities)](const double r) {
        std::size_t index = static_cast<std::size_t>(
            std::upper_bound(r_ends.begin(), r_ends.end(), r) -
            r_ends.begin());
        index = std::min(index, densities.size() - 1);
        return densities[index];
      };
    }

    const tenryu::core::ZoningResult result =
        tenryu::core::compute_zoning_intent_nodes(
            mesh.r_min, mesh.r_max, zic, rho0);
    if (!result.ok) {
      throw ConfigError("[mesh-zoning-intent] " + result.diag.code + ": " +
                        result.diag.message);
    }
    mesh.explicit_nodes = result.nodes;
    const int n = static_cast<int>(mesh.explicit_nodes.size()) - 1;
    if (mesh.nr >= 0 && mesh.nr != n) {
      throw ConfigError(
          "Mesh.nr (" + std::to_string(mesh.nr) +
          ") conflicts with Mesh.zoning_intent n_cells (" +
          std::to_string(n) + "); omit Mesh.nr or match it");
    }
    mesh.nr = n;
    tenryu::core::log_info(
        "[mesh-zoning-intent] nodes=" +
        std::to_string(mesh.explicit_nodes.size()) +
        " measure=" + zoning.measure +
        " ratio_max=" + std::to_string(result.diag.ratio_max_achieved) +
        " ratio_mean=" + std::to_string(result.diag.ratio_mean_achieved) +
        " soft_exceed=" + std::to_string(result.diag.n_ratio_soft_exceed) +
        " width_min=" + std::to_string(result.diag.width_min_achieved) +
        " quad_residual=" +
        std::to_string(result.diag.quadrature_rel_residual));
    for (const auto& w : result.diag.warnings) {
      tenryu::core::log_warning("[mesh-zoning-intent] " + w);
    }
  }
  if (mesh.nr < 4 && mesh.logical_mesh_2d != "cone_shell") {
    throw ValueError("Mesh.nr must be >= 4");
  }
  if (mesh.logical_mesh_2d != "rectangular_rz" &&
      mesh.logical_mesh_2d != "spherical_polar_halfplane" &&
      mesh.logical_mesh_2d != "polar_in_box" &&
      mesh.logical_mesh_2d != "cone_shell") {
    throw ValueError(
        "Mesh.logical_mesh_2d must be one of {\"rectangular_rz\", "
        "\"spherical_polar_halfplane\", \"polar_in_box\", "
        "\"cone_shell\"}");
  }
  if (mesh.polar_center_treatment != "annular" &&
      mesh.polar_center_treatment != "tri_fan" &&
      mesh.polar_center_treatment != "button") {
    throw ValueError(
        "Mesh.polar_center_treatment must be one of {\"annular\", \"tri_fan\", \"button\"}");
  }
  if ((mesh.polar_center_treatment == "tri_fan" ||
       mesh.polar_center_treatment == "button") &&
      !tenryu::core::is_polar_family(mesh.logical_mesh_2d)) {
    throw ConfigError(
        "Mesh.polar_center_treatment='" + mesh.polar_center_treatment + "' requires "
        "a polar-family Mesh.logical_mesh_2d");
  }
  if (!(mesh.spherical_polar_s_max > 0.0)) {
    throw ValueError("Mesh.spherical_polar_s_max must be > 0");
  }
  if (mesh.logical_mesh_2d == "polar_in_box") {
    const double s_max_tol =
        1.0e-12 * std::max(1.0, std::abs(mesh.spherical_polar_s_max));
    if (!(std::abs(mesh.explicit_nodes.back() -
                   mesh.spherical_polar_s_max) <= s_max_tol)) {
      throw ConfigError(
          "polar_in_box Mesh.explicit_nodes must end at "
          "Mesh.spherical_polar_s_max");
    }
    if (!(std::isfinite(mesh.box_r_max) && mesh.box_r_max > 0.0)) {
      throw ValueError("polar_in_box Mesh.box_r_max must be finite and > 0");
    }
    if (!(std::isfinite(mesh.box_z_min) &&
          std::isfinite(mesh.box_z_max) &&
          mesh.box_z_max > mesh.box_z_min)) {
      throw ValueError(
          "polar_in_box Mesh.box_z_min/box_z_max must be finite with "
          "box_z_max > box_z_min");
    }
    const auto extent_matches = [](const double value,
                                   const double derived) {
      return std::isfinite(value) &&
             std::abs(value - derived) <=
                 1.0e-12 * std::max(std::abs(value), std::abs(derived));
    };
    if ((mesh_r_min_explicit && !extent_matches(mesh.r_min, 0.0)) ||
        (mesh_r_max_explicit &&
         !extent_matches(mesh.r_max, mesh.box_r_max)) ||
        (mesh_z_min_explicit &&
         !extent_matches(mesh.z_min, mesh.box_z_min)) ||
        (mesh_z_max_explicit &&
         !extent_matches(mesh.z_max, mesh.box_z_max))) {
      throw ConfigError(
          "polar_in_box derives r/z extents from the box; remove or match the explicit values");
    }
    mesh.r_min = 0.0;
    mesh.r_max = mesh.box_r_max;
    mesh.z_min = mesh.box_z_min;
    mesh.z_max = mesh.box_z_max;
    if (!(std::isfinite(mesh.box_center_z) &&
          mesh.box_center_z > mesh.box_z_min &&
          mesh.box_center_z < mesh.box_z_max)) {
      throw ValueError(
          "polar_in_box Mesh.box_center_z must be finite and strictly inside "
          "the box z bounds");
    }
    if (!(mesh.box_r_max > mesh.spherical_polar_s_max &&
          mesh.box_z_max >
              mesh.box_center_z + mesh.spherical_polar_s_max &&
          mesh.box_z_min <
              mesh.box_center_z - mesh.spherical_polar_s_max)) {
      throw ConfigError(
          "polar_in_box box must strictly contain the polar prefix (s_max)");
    }
    if (mesh.morph_rings < 4) {
      throw ValueError("Mesh.morph_rings must be >= 4 for polar_in_box");
    }
    if (mesh.collar_rings < 2 || mesh.collar_rings > 32) {
      throw ValueError(
          "Mesh.collar_rings must be in [2, 32] for polar_in_box");
    }
    if (!(std::isfinite(mesh.morph_growth_max) &&
          mesh.morph_growth_max > 1.0 &&
          mesh.morph_growth_max <= 2.0)) {
      throw ValueError(
          "Mesh.morph_growth_max must be finite and in (1, 2] for "
          "polar_in_box");
    }
  }
  if (std::isfinite(mesh.cone_theta_wall)) {
    constexpr double kPiCone = 3.14159265358979323846;
    if (mesh.logical_mesh_2d != "polar_in_box") {
      throw ConfigError(
          "Mesh.cone_theta_wall requires Mesh.logical_mesh_2d=\"polar_in_box\"");
    }
    if (!(mesh.cone_theta_wall > 0.0 &&
          mesh.cone_theta_wall < kPiCone)) {
      throw ConfigError("Mesh.cone_theta_wall must be in (0, pi)");
    }
    if (!(std::isfinite(mesh.cone_tip_radius) &&
          mesh.cone_tip_radius > mesh.explicit_nodes.back())) {
      throw ConfigError(
          "Mesh.cone_tip_radius must be finite and greater than the polar "
          "lock radius (Mesh.explicit_nodes.back())");
    }
    if (!(std::isfinite(mesh.cone_activation_radius) &&
          mesh.cone_activation_radius >= mesh.explicit_nodes.back() &&
          mesh.cone_activation_radius < mesh.cone_tip_radius)) {
      throw ConfigError(
          "Mesh.cone_activation_radius must be finite, at least the polar "
          "lock radius, and less than Mesh.cone_tip_radius");
    }
    if (mesh.cone_fine_cells_minus < 1 ||
        mesh.cone_fine_cells_plus < 1) {
      throw ConfigError(
          "Mesh.cone_fine_cells_minus and Mesh.cone_fine_cells_plus must "
          "each be >= 1");
    }
    if (!(std::isfinite(mesh.cone_angular_growth_max) &&
          mesh.cone_angular_growth_max > 1.0 &&
          mesh.cone_angular_growth_max <= 2.0)) {
      throw ConfigError(
          "Mesh.cone_angular_growth_max must be finite and in (1, 2]");
    }
    if (mesh.cone_tip_style != "single_line") {
      throw ConfigError("cone_tip_style v1 supports single_line only");
    }
    const std::int64_t cone_angular_cell_budget =
        static_cast<std::int64_t>(mesh.cone_fine_cells_minus) +
        static_cast<std::int64_t>(mesh.cone_fine_cells_plus) + 2 * 4 +
        2 * 2 + 2 * 1;
    if (cone_angular_cell_budget > static_cast<std::int64_t>(mesh.nz)) {
      throw ConfigError(
          "Mesh cone angular capacity budget violated: "
          "cone_fine_cells_minus + cone_fine_cells_plus + "
          "2*4 (min transition per side) + 2*2 (min corner separations) + "
          "2*1 (axis neighborhoods) must be <= nz");
    }
    constexpr bool kPolarInBoxConeGenerationAvailable = true;
    if (!kPolarInBoxConeGenerationAvailable) {
      throw ConfigError("polar_in_box cone generation lands with B1b");
    }
  }
  if (mesh.polar_center_treatment != "tri_fan" &&
      !(mesh.spherical_polar_kappa > 0.0)) {
    throw ValueError("Mesh.spherical_polar_kappa must be > 0");
  }
  if (mesh.multiblock_cart_core_bridge_grading == "quintic_log" &&
      mesh.multiblock_transition_scheme !=
          MultiblockTransitionScheme::HERMITE_BRIDGE) {
    throw ConfigError(
        "multiblock_cart_core_bridge_grading=\"quintic_log\" requires "
        "multiblock_transition_scheme=\"hermite_bridge\"");
  }
  tenryu::core::validate_multiblock_topology_config(config);
  if (numerics.diagnostics.refinement_autopilot.mode == "arm_exit" &&
      mesh.topology_scheme == TopologyScheme::SINGLE_BLOCK) {
    throw ConfigError(
        "Diagnostics.refinement_autopilot.mode=\"arm_exit\" requires a "
        "multiblock topology (the epoch/swap machinery is multiblock-only; "
        "use \"shadow\" on single-block meshes)");
  }
  if (numerics.conduction.enabled && main.dimension == "2D_RZ" &&
      mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK) {
    throw ConfigError(
        "Numerics.conduction.enabled=True requires the single_block 2D_RZ "
        "topology (the structured conduction kernels do not address "
        "multiblock CSR meshes; fail-closed refusal)");
  }
  if (mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK &&
      mesh.topology_scheme != TopologyScheme::PENTAGON_BELT_SHELL) {
    if (!mesh.grid_segments.empty()) {
      const double r_match_ref = mesh.multiblock_cart_core_r_match;
      const double s_max_ref = mesh.spherical_polar_s_max;
      const double r_match_tol =
          1.0e-12 * std::max(1.0, std::abs(r_match_ref));
      const double s_max_tol = 1.0e-12 * std::max(1.0, std::abs(s_max_ref));
      if (!(std::abs(mesh.grid_segments.front().r_start - r_match_ref) <=
            r_match_tol)) {
        throw ConfigError(
            "multiblock shell grid segments must start at "
            "multiblock_cart_core_r_match (the shell inner radius)");
      }
      if (!(std::abs(mesh.grid_segments.back().r_end - s_max_ref) <=
            s_max_tol)) {
        throw ConfigError(
            "multiblock shell grid segments must end at "
            "spherical_polar_s_max");
      }
      int n_total = 0;
      for (const auto& segment : mesh.grid_segments) {
        n_total += segment.nr;
      }
      if (n_total != mesh.nr) {
        throw ConfigError(
            "multiblock shell grid segments total zones (" +
            std::to_string(n_total) + ") must equal Mesh.nr (" +
            std::to_string(mesh.nr) + ", the shell radial count)");
      }
    }
  }
  if (mesh.logical_mesh_2d == "spherical_polar_halfplane") {
    if (radiation.enabled) {
      throw ConfigError(
          "Mesh.logical_mesh_2d='spherical_polar_halfplane' is hydro-only "
          "(Phase 6-minimum); Radiation must be disabled");
    }
    if (laser.enabled) {
      throw ConfigError(
          "Mesh.logical_mesh_2d='spherical_polar_halfplane' is hydro-only; "
          "Laser must be disabled");
    }
    if (numerics.conduction.enabled) {
      throw ConfigError(
          "Mesh.logical_mesh_2d='spherical_polar_halfplane' is hydro-only; "
          "Numerics.conduction must be disabled");
    }
  }

  if (is_dimension_1d(main.dimension)) {
    if (mesh.logical_mesh_2d != "rectangular_rz") {
      throw ConfigError("Mesh.logical_mesh_2d is valid only for 2D_RZ");
    }
    if (mesh.nz != 1) {
      throw ConfigError("nz is not valid for 1D_SPH");
    }
    if (mesh.r_min > 0.0) {
      tenryu::core::log_warning(
          "1D_SPH with Mesh.r_min>0 excludes r=0; full-center symmetry is not represented");
    }
    if (mesh.motion == "ale") {
      throw ConfigError("ALE not supported for 1D_SPH");
    }
    if (!motion_explicitly_set) {
      mesh.motion = "lagrangian";
    }
  } else {
    if (mesh.nz < 4) {
      throw ConfigError("2D_RZ requires nz >= 4");
    }
    if (mesh.logical_mesh_2d == "rectangular_rz" &&
        (std::isnan(mesh.z_min) || std::isnan(mesh.z_max) || !(mesh.z_max > mesh.z_min))) {
      throw ConfigError("2D_RZ requires explicit r_min, r_max, z_min, z_max");
    }
    if (!(numerics.axis_eps_cm > 0.0) || !std::isfinite(numerics.axis_eps_cm)) {
      throw ConfigError("Numerics.axis_eps_cm must be finite and > 0");
    }
    const bool r_min_is_axis = std::abs(mesh.r_min) <= numerics.axis_eps_cm;
    if (mesh.logical_mesh_2d == "rectangular_rz" && !r_min_is_axis) {
      // H-05: Annular geometry (r_min > 0) is valid for problems without axis BC.
      // Axis boundary kernel at boundary_2d.cu applies only when i==0 nodes
      // are at r~0.  Warn, but do not reject — verified tests use annular grids.
      core::log_warning("2D_RZ with r_min > 0: axis boundary conditions assume r_min == 0; "
                        "annular geometry disables axis symmetry enforcement");
    }
    if (mesh.logical_mesh_2d == "rectangular_rz" && r_min_is_axis) {
      mesh.r_min = 0.0;
    }
    if (!motion_explicitly_set) {
      mesh.motion = "ale";
    }
  }

  if (materials.materials.empty()) {
    throw ConfigError("Materials requires at least one Material definition");
  }
  if (materials.materials.front().is_void) {
    throw ConfigError("Materials.materials[0] must not be a void material");
  }
  if (materials.first_nonvoid_material_index() < 0) {
    throw ConfigError("Materials must include at least one non-void material");
  }
  std::string first_nlte_file;    // tracks first NLTE tabular opacity file for group derivation
  std::string first_nlte_source;  // "IONMIX" or "TMAT"
  std::vector<double> first_nlte_original_bounds;
  int tmat_ionization_count = 0;
  std::set<std::string> names;
  for (std::size_t material_index = 0;
       material_index < materials.materials.size();
       ++material_index) {
    auto& mat = materials.materials[material_index];
    if (mat.name.empty()) {
      throw ConfigError("Materials.materials[].name must not be empty");
    }
    if (!names.insert(mat.name).second) {
      throw ConfigError("Duplicate material name: " + mat.name);
    }
    if (!(mat.A > 0.0)) {
      throw ConfigError("Materials.materials[\"" + mat.name + "\"].A must be > 0");
    }
    if (!(mat.Z >= 0.0)) {
      throw ConfigError("Materials.materials[\"" + mat.name + "\"].Z must be >= 0");
    }
    if (!(mat.ideal_gas_gamma > 1.0)) {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.ideal_gas.gamma must be > 1");
    }
    if (numerics.hydro.compatible_energy && !mat.is_void) {
      const bool ideal_supported =
          mat.eos_model == "ideal_gas" &&
          (mat.hydro_eos_backend == "legacy" ||
           mat.hydro_eos_backend == "exact_ideal_gas");
      const bool tmat_supported =
          mat.eos_model == "tmat" && mat.hydro_eos_backend == "legacy";
      if (!ideal_supported && !tmat_supported) {
        throw ConfigError(
            "Numerics.hydro.compatible_energy=True requires ideal_gas EOS "
            "or tmat EOS with eos.hydro_backend=\"legacy\"; material \"" +
            mat.name + "\" uses eos.model=\"" + mat.eos_model +
            "\" and eos.hydro_backend=\"" + mat.hydro_eos_backend + "\"");
      }
    }
    if (mat.hydro_eos_backend == "exact_ideal_gas" && main.dim != 1) {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.hydro_backend=\"exact_ideal_gas\" is supported only "
                        "for Main.dimension=\"1D_SPH\"");
    }
    if (mat.hydro_eos_backend == "rho_e_table" && main.dim != 1) {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.hydro_backend=\"rho_e_table\" is supported only "
                        "for Main.dimension=\"1D_SPH\"");
    }
    if (mat.hydro_eos_backend == "mie_gruneisen" && main.dim != 1) {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.hydro_backend=\"mie_gruneisen\" is supported only "
                        "for Main.dimension=\"1D_SPH\"");
    }
    if (mat.hydro_eos_backend == "mie_gruneisen" && !main.two_temperature) {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.hydro_backend=\"mie_gruneisen\" requires 2T hydro");
    }
    if (mat.hydro_eos_backend == "mie_gruneisen" && mat.eos_model == "ideal_gas") {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.hydro_backend=\"mie_gruneisen\" requires tabular EOS");
    }
    if (mat.hydro_eos_backend == "mie_gruneisen" && !(mat.mg_T_ref_eV > 0.0)) {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.mg_T_ref_eV must be > 0 for mie_gruneisen backend");
    }
    if (mat.hydro_eos_backend == "mie_gruneisen" && !(mat.mg_dT_rel > 0.0)) {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.mg_dT_rel must be > 0 for mie_gruneisen backend");
    }
    if (mat.is_void) {
      if (!mat.eos_file.empty()) {
        tenryu::core::log_warning("Materials.materials[\"" + mat.name +
                                  "\"].eos.file is ignored because is_void=true");
      }
      if (!mat.opacity_file.empty()) {
        tenryu::core::log_warning("Materials.materials[\"" + mat.name +
                                  "\"].opacity.file is ignored because is_void=true");
      }
    }
    if (!mat.is_void && mat.eos_model == "ionmix") {
      if (mat.eos_file.empty()) {
        throw ConfigError("eos.file is required for eos.model='ionmix'");
      }
      materials::IonmixEOSData eos_data;
      try {
        eos_data = materials::load_ionmix_binary_eos(mat.eos_file);
        const auto validated_zbar = materials::ionmix_eos_to_zbar_table(eos_data, mat.A);
        (void)validated_zbar;
      } catch (const std::exception& ex) {
        throw ConfigError("Failed to load IONMIX EOS file for Materials.materials[\"" +
                          mat.name + "\"].eos.file=\"" + mat.eos_file + "\": " +
                          ex.what());
      }
      mat.eos_tables = std::make_shared<const materials::EOSTableTriplet>(
          materials::ionmix_eos_to_table_triplet(eos_data, mat.A));
      tenryu::core::log_info("IONMIX EOS loaded for material '" + mat.name + "': " +
                             std::to_string(eos_data.ndens) + " x " +
                             std::to_string(eos_data.ntemp) +
                             " grid, table EOS active.");
    }
    if (!mat.is_void && mat.eos_model == "tmat") {
      if (mat.eos_file.empty()) {
        throw ConfigError("eos.file is required for eos.model='tmat'");
      }
      materials::TmatFile tmat;
      try {
        tmat = materials::load_tmat(mat.eos_file);
      } catch (const std::exception& ex) {
        throw ConfigError("Failed to load TMAT file for Materials.materials[\"" +
                          mat.name + "\"].eos.file=\"" + mat.eos_file + "\": " +
                          ex.what());
      }
      if (!tmat.eos.has_value()) {
        throw ConfigError("TMAT file \"" + mat.eos_file +
                          "\" has eos.model='tmat' but does not contain /eos payload");
      }
      if (tmat.ionization.has_value()) {
        ++tmat_ionization_count;
        if (tmat_ionization_count == 1) {
          const materials::ZeffRatioTable zeff_table =
              materials::resample_zeff_ratio_log_uniform(
                  materials::tmat_ionization_to_zeff_ratio(
                      *tmat.ionization, tmat.material),
                  256,
                  128);
          materials.zmoments.ndens = zeff_table.ndens;
          materials.zmoments.ntemp = zeff_table.ntemp;
          materials.zmoments.ni_grid = zeff_table.ni_grid;
          materials.zmoments.T_grid_eV = zeff_table.T_grid_eV;
          materials.zmoments.r2 = zeff_table.ratio;
          materials.zmoments.r4 = zeff_table.ratio4;
          materials.zmoments.provider_material =
              static_cast<int>(material_index);
          // Laser retains its independent copy; shared hydro/laser moments
          // are stored in MaterialsConfig::zmoments above.
          laser.ib.zeff_table.ndens = zeff_table.ndens;
          laser.ib.zeff_table.ntemp = zeff_table.ntemp;
          laser.ib.zeff_table.ni_grid = zeff_table.ni_grid;
          laser.ib.zeff_table.T_grid_eV = zeff_table.T_grid_eV;
          laser.ib.zeff_table.ratio = zeff_table.ratio;
        }
      }
      mat.eos_tables = std::make_shared<const materials::EOSTableTriplet>(
          materials::tmat_eos_to_table_triplet(*tmat.eos, mat.A));
      const auto validated_zbar = materials::tmat_eos_to_zbar_table(*tmat.eos, mat.A);
      (void)validated_zbar;
      tenryu::core::log_info("TMAT EOS loaded for material '" + mat.name + "': " +
                             std::to_string(tmat.eos->ndens) + " x " +
                             std::to_string(tmat.eos->ntemp) +
                             " grid, table EOS active.");
    }
    if (!mat.is_void && mat.eos_model == "power_law_te") {
      mat.eos_tables = std::make_shared<const materials::EOSTableTriplet>(
          materials::build_power_law_te_tables(
              mat.A,
              mat.ideal_gas_gamma,
              mat.eos_power_law_f_erg_g,
              mat.eos_power_law_beta,
              mat.eos_power_law_mu_rho,
              mat.eos_power_law_gamma_p,
              mat.eos_power_law_step_D_erg_g_eV,
              mat.eos_power_law_step_Tc_eV,
              mat.eos_power_law_step_w_eV));
      tenryu::core::log_info("power_law_te EOS tabulated for material '" + mat.name +
                             "': 64 x 512 grid, table EOS active.");
    }
    if (!mat.is_void && mat.eos_model == "sesame" && mat.eos_file.empty()) {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.file is required for model=" + mat.eos_model);
    }
    if (!mat.is_void && mat.eos_model == "sesame" && mat.sesame_material_id < 0) {
      throw ConfigError("Materials.materials[\"" + mat.name +
                        "\"].eos.sesame_material_id is required for model=sesame");
    }
    if (!mat.is_void && mat.eos_model == "sesame") {
      // load_sesame returns {total, electron}; 301/304 grids may differ, so the
      // ion table resamples 304 onto the 301 grid before differencing
      // (build_sesame_ion_table) — node-wise subtraction across differently
      // shaped arrays would be out-of-bounds.
      auto [total, electron] = materials::load_sesame(mat.eos_file, mat.sesame_material_id);
      materials::EOSTable ion = materials::build_sesame_ion_table(total, electron);
      mat.eos_tables = std::make_shared<const materials::EOSTableTriplet>(
          materials::EOSTableTriplet{std::move(ion), std::move(electron), std::move(total)});
      tenryu::core::log_info("SESAME EOS loaded for material '" + mat.name +
                             "', table EOS active.");
    }
    mat.eos_signature = compute_material_eos_signature(mat);
    if (!mat.is_void && materials.zbar.model == "tabular") {
      std::string zbar_source_file;
      if ((mat.eos_model == "ionmix" || mat.eos_model == "tmat") &&
          !mat.eos_file.empty()) {
        zbar_source_file = mat.eos_file;
      } else if ((mat.opacity_model == "table_nlte" || mat.opacity_model == "ionmix" ||
                  mat.opacity_model == "tmat") &&
                 !mat.opacity_file.empty()) {
        zbar_source_file = mat.opacity_file;
      } else if (!materials.zbar.table_file.empty()) {
        zbar_source_file = materials.zbar.table_file;
      }
      if (zbar_source_file.empty()) {
        throw ConfigError("zbar.model='tabular' for material '" + mat.name +
                          "' requires tabular source: eos.model in {'ionmix','tmat'}, "
                          "opacity.model in {'ionmix','table_nlte','tmat'}, or "
                          "zbar.table_file");
      }
    }
    if (!mat.is_void &&
        (mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat")) {
      if (mat.opacity_file.empty()) {
        throw ConfigError("Materials.materials[\"" + mat.name +
                          "\"].opacity.file is required for model=" + mat.opacity_model);
      }
      if (!is_lambda_method(mat.lambda_method)) {
        throw ConfigError("Materials.materials[\"" + mat.name +
                          "\"].opacity.lambda_method must be one of "
                          "{\"finite_difference\", \"freeze_opacity\"}");
      }
      if (!(mat.lambda_fd_delta_rel > 0.0)) {
        throw ValueError("Materials.materials[\"" + mat.name +
                         "\"].opacity.lambda_fd_delta_rel must be > 0");
      }
      if (!(mat.lambda_fd_abs_min > 0.0)) {
        throw ValueError("Materials.materials[\"" + mat.name +
                         "\"].opacity.lambda_fd_abs_min must be > 0");
      }
      if (!(mat.nlte_f_min > 0.0 && mat.nlte_f_min <= 1.0)) {
        throw ValueError("Materials.materials[\"" + mat.name +
                         "\"].opacity.f_min must be in (0, 1]");
      }

      materials::IonmixOpacityData ionmix_opacity;
      if (mat.opacity_model == "table_nlte") {
        try {
          ionmix_opacity = materials::load_ionmix_opacity(mat.opacity_file);
        } catch (const std::exception& ex) {
          throw ConfigError("Failed to load IONMIX opacity file for "
                            "Materials.materials[\"" +
                            mat.name + "\"].opacity.file=\"" + mat.opacity_file +
                            "\": " + ex.what());
        }
      } else {
        try {
          const materials::TmatFile tmat = materials::load_tmat(mat.opacity_file);
          if (!tmat.opacity.has_value()) {
            throw ConfigError("TMAT file \"" + mat.opacity_file +
                              "\" has opacity.model='tmat' but does not contain "
                              "/opacity payload");
          }
          ionmix_opacity = materials::tmat_to_ionmix_opacity(
              *tmat.opacity, mat.tmat_skip_lte_repair);
        } catch (const std::exception& ex) {
          throw ConfigError("Failed to load TMAT opacity file for "
                            "Materials.materials[\"" +
                            mat.name + "\"].opacity.file=\"" + mat.opacity_file +
                            "\": " + ex.what());
        }
      }
      const std::vector<double> opacity_T_range =
          opacity_temperature_range(ionmix_opacity);
      if (valid_temperature_range(opacity_T_range)) {
        expand_temperature_range(radiation.opacity_T_range_eV,
                                 opacity_T_range[0],
                                 opacity_T_range[1]);
      }

      if (first_nlte_file.empty()) {
        // First table-driven NLTE material: adopt group structure from file.
        first_nlte_file = mat.opacity_file;
        first_nlte_source = (mat.opacity_model == "tmat") ? "TMAT" : "IONMIX";
        first_nlte_original_bounds = ionmix_opacity.bounds_eV;
        radiation.groups = ionmix_opacity.ngroups;
        if (radiation.group_repack_hard_xray) {
          radiation.group_bounds_eV =
              tenryu::core::repack_radiation_group_bounds_for_hard_xray(
                  ionmix_opacity.ngroups,
                  repack_energy_range(radiation, ionmix_opacity));
          if (radiation.group_bounds_eV.empty()) {
            throw ConfigError(
                "Radiation.group_repack_hard_xray failed to produce group bounds");
          }
          const int hard_xray_groups =
              tenryu::core::count_radiation_groups_inside_energy_band(
                  radiation.group_bounds_eV, 2000.0, 5000.0);
          if (hard_xray_groups < 20) {
            throw ConfigError(
                "Radiation.group_repack_hard_xray produced fewer than 20 groups "
                "inside 2-5 keV");
          }
        } else {
          radiation.group_bounds_eV = ionmix_opacity.bounds_eV;
        }
        tenryu::core::log_info("Radiation group structure derived from " +
                               first_nlte_source + " file \"" + mat.opacity_file + "\": " +
                               std::to_string(ionmix_opacity.ngroups) + " groups");
      } else {
        // Subsequent table-driven NLTE material: must match already-set groups.
        if (ionmix_opacity.ngroups != radiation.groups) {
          throw ConfigError("Tabular opacity group count mismatch: \"" +
                            mat.opacity_file + "\" has " +
                            std::to_string(ionmix_opacity.ngroups) + " groups, but \"" +
                            first_nlte_file + "\" has " +
                            std::to_string(radiation.groups) + " groups");
        }
        for (std::size_t g = 0; g < radiation.group_bounds_eV.size(); ++g) {
          const double ref_bound = first_nlte_original_bounds[g];
          const double file_bound = ionmix_opacity.bounds_eV[g];
          const double denom =
              std::max({std::abs(ref_bound), std::abs(file_bound), 1.0e-30});
          const double rel_err = std::abs(ref_bound - file_bound) / denom;
          if (rel_err > 1.0e-6) {
            throw ConfigError("Tabular opacity group boundary mismatch between \"" +
                              first_nlte_file + "\" and \"" + mat.opacity_file +
                              "\" at index " + std::to_string(g) + ": " +
                              std::to_string(ref_bound) + " vs " +
                              std::to_string(file_bound) +
                              ", rel_err=" + std::to_string(rel_err) +
                              " (tol=1e-6)");
          }
        }
      }
    }
    if (!is_runtime_supported_opacity_model(mat.opacity_model)) {
      tenryu::core::log_warning(
          "Materials.materials[\"" + mat.name + "\"].opacity.model=\"" +
          mat.opacity_model +
          "\" requests tabular opacity, but IONMIX/SESAME tabular opacity "
          "interpolation is not implemented yet; current runtime path would fall "
          "back to constant opacity.");
      throw ConfigError(
          "Materials.materials[\"" + mat.name + "\"].opacity.model=\"" +
          mat.opacity_model +
          "\" is not supported in this build. Use opacity.model=\"constant\" or "
          "\"freq_dep_marshak\" or \"table_nlte\" or \"tmat\" or "
          "\"power_law\" explicitly.");
    }
  }

  materials.zbar_tables.clear();
  if (materials.zbar.model == "tabular") {
    materials.zbar_tables.reserve(materials.materials.size());
    for (const auto& mat : materials.materials) {
      if (mat.is_void) {
        materials.zbar_tables.push_back(
            std::make_shared<const materials::IonmixZbarTable>(make_zero_zbar_table()));
        continue;
      }
      std::string zbar_source;
      bool zbar_source_is_tmat = false;
      if ((mat.eos_model == "ionmix" || mat.eos_model == "tmat") &&
          !mat.eos_file.empty()) {
        zbar_source = mat.eos_file;
        zbar_source_is_tmat = (mat.eos_model == "tmat");
      } else if ((mat.opacity_model == "table_nlte" || mat.opacity_model == "ionmix" ||
                  mat.opacity_model == "tmat") &&
                 !mat.opacity_file.empty()) {
        zbar_source = mat.opacity_file;
        zbar_source_is_tmat = (mat.opacity_model == "tmat");
      } else if (!materials.zbar.table_file.empty()) {
        zbar_source = materials.zbar.table_file;
        zbar_source_is_tmat = zbar_source.ends_with(".tmat.h5");
      }

      TENRYU_ASSERT(!zbar_source.empty(),
                    "tabular Zbar source must be set after validation");

      try {
        materials::IonmixZbarTable table;
        if (zbar_source_is_tmat) {
          const materials::TmatFile tmat = materials::load_tmat(zbar_source);
          if (!tmat.eos.has_value()) {
            throw ConfigError("TMAT source '" + zbar_source +
                              "' for zbar.model='tabular' does not contain /eos payload");
          }
          table = materials::tmat_eos_to_zbar_table(*tmat.eos, mat.A);
        } else {
          const materials::IonmixEOSData eos_data =
              materials::load_ionmix_binary_eos(zbar_source);
          table = materials::ionmix_eos_to_zbar_table(eos_data, mat.A);
        }
        materials.zbar_tables.push_back(
            std::make_shared<const materials::IonmixZbarTable>(std::move(table)));
      } catch (const std::exception& ex) {
        throw ConfigError("Failed to build tabular Zbar table for material '" + mat.name +
                          "' from source '" + zbar_source + "': " + ex.what());
      }
    }
  }

  if (!geometry.rho.detected) {
    throw ConfigError("Geometry.rho must be callable");
  }
  if (!geometry.Te.detected) {
    throw ConfigError("Geometry.Te must be callable");
  }
  if (!geometry.Ti.detected) {
    throw ConfigError("Geometry.Ti must be callable");
  }
  if (geometry.volfrac.empty()) {
    throw ConfigError("Geometry.volfrac must define one callable per material");
  }
  for (const auto& mat : materials.materials) {
    if (!geometry.volfrac.contains(mat.name)) {
      throw ConfigError("Geometry.volfrac is missing material key: " + mat.name);
    }
  }
  for (const auto& [name, _] : geometry.volfrac) {
    if (!names.contains(name)) {
      throw ConfigError("Geometry.volfrac contains unknown material key: " + name);
    }
  }

  // Auto-set 1-group grey radiation when all materials use constant opacity
  // and no group bounds were explicitly specified or derived from tables.
  if (radiation.enabled && radiation.group_bounds_eV.empty()) {
    bool all_constant = true;
    for (const auto& mat : materials.materials) {
      if (mat.opacity_model != "constant") {
        all_constant = false;
        break;
      }
    }
    if (all_constant) {
      radiation.groups = 1;
      radiation.group_bounds_eV = {0.0, 1.0e6};
      tenryu::core::log_info(
          "All materials use constant opacity; auto-set 1-group grey radiation [0, 1e6 eV]");
    }
  }

  if (radiation.groups < 1) {
    throw ValueError("Radiation.groups must be >= 1");
  }
  {
    int radiation_group_count = radiation.groups;
    if (!radiation.group_bounds_eV.empty()) {
      radiation_group_count = static_cast<int>(radiation.group_bounds_eV.size()) - 1;
    }
    const bool any_power_law_opacity =
        std::any_of(materials.materials.begin(),
                    materials.materials.end(),
                    [](const auto& mat) { return mat.opacity_model == "power_law"; });
    if (any_power_law_opacity && radiation_group_count != 1) {
      throw ConfigError(
          "opacity.model=\"power_law\" is grey-only (v1): requires exactly 1 radiation group");
    }
  }
  {
    const bool any_freq_dep_opacity =
        std::any_of(materials.materials.begin(),
                    materials.materials.end(),
                    [](const auto& mat) { return mat.opacity_model == "freq_dep_marshak"; });
    if (any_freq_dep_opacity && !radiation.group_bounds_eV.empty() &&
        !(radiation.group_bounds_eV.front() > 0.0)) {
      throw ConfigError(
          "opacity.model=\"freq_dep_marshak\" requires group_bounds_eV[0] > 0: the "
          "group-mean integrals start at the first bound (the host integrator "
          "returns zero and the device integrates an unresolved [1e-300, E_1] "
          "log span when E_0 <= 0)");
    }
  }
  if (!(radiation.imc.alpha > 0.0)) {
    throw ValueError("Radiation.imc.alpha must be > 0");
  }
  if (radiation.imc.particles_per_cell_group < 1) {
    throw ConfigError("Radiation.imc.particles_per_cell_group must be >= 1");
  }
  if (radiation.imc.particle_budget > 0 &&
      radiation.imc.particle_budget < radiation.imc.particles_per_cell_group) {
    throw ConfigError(
        "Radiation.imc.particle_budget must be >= particles_per_cell_group or -1 (disabled)");
  }
  if (!(radiation.imc.difference.W_max >= 0.0 &&
        radiation.imc.difference.W_max <= 1.0)) {
    throw ValueError("Radiation.imc.difference.W_max must be in [0, 1]");
  }
  if (!(radiation.imc.difference.tau0 > 0.0)) {
    throw ValueError("Radiation.imc.difference.tau0 must be > 0");
  }
  if (!(radiation.imc.difference.chi0 > 0.0)) {
    throw ValueError("Radiation.imc.difference.chi0 must be > 0");
  }
  if (radiation.imc.difference.enabled) {
    if (main.dimension == "2D_RZ" && radiation.imc.difference.face_transport) {
      throw ConfigError(
          "Radiation.imc.difference.face_transport=True is not yet supported for "
          "Main.dimension=\"2D_RZ\". Use face_transport=False.");
    }
    if (main.dimension != "1D_SPH" && main.dimension != "2D_RZ") {
      throw ConfigError(
          "Radiation.imc.difference.enabled currently requires "
          "Main.dimension=\"1D_SPH\" or \"2D_RZ\"");
    }
  }
  if (!(radiation.imc.spectral_bias_eta >= 0.0 &&
        radiation.imc.spectral_bias_eta <= 1.0)) {
    throw ValueError("Radiation.imc.spectral_bias_eta must be in [0, 1]");
  }
  if (!(radiation.imc.sloc_ema_beta >= 0.0 &&
        radiation.imc.sloc_ema_beta <= 1.0)) {
    throw ValueError("Radiation.imc.sloc_ema_beta must be in [0, 1]");
  }
  if (!(radiation.imc.sloc_sigma_floor > 0.0)) {
    throw ValueError("Radiation.imc.sloc_sigma_floor must be > 0");
  }
  if (!(radiation.imc.sloc_sigma_cap > 0.0)) {
    throw ValueError("Radiation.imc.sloc_sigma_cap must be > 0");
  }
  if (radiation.imc.sloc_sigma_floor > radiation.imc.sloc_sigma_cap) {
    throw ValueError(
        "Radiation.imc.sloc_sigma_floor must be <= Radiation.imc.sloc_sigma_cap");
  }
  if (!(radiation.imc.sloc_tau_ref > 0.0)) {
    throw ValueError("Radiation.imc.sloc_tau_ref must be > 0");
  }
  const double net_e_source_smoothing_alpha_max =
      (main.dimension == "2D_RZ" &&
       radiation.imc.net_e_source_smoothing.enabled)
          ? 0.125
          : 0.25;
  if (!(radiation.imc.net_e_source_smoothing.alpha >= 0.0 &&
        radiation.imc.net_e_source_smoothing.alpha <=
            net_e_source_smoothing_alpha_max)) {
    throw ValueError(
        "Radiation.imc.net_e_source_smoothing.alpha must be in [0, " +
        std::to_string(net_e_source_smoothing_alpha_max) + "]");
  }
  if (!(radiation.imc.net_e_source_smoothing.tau_threshold > 0.0)) {
    throw ValueError(
        "Radiation.imc.net_e_source_smoothing.tau_threshold must be > 0");
  }
  if (radiation.imc.net_e_source_smoothing.passes < 0) {
    throw ValueError("Radiation.imc.net_e_source_smoothing.passes must be >= 0");
  }
  if (!(radiation.imc.net_e_source_smoothing.grad_Te_scale > 0.0)) {
    throw ValueError(
        "Radiation.imc.net_e_source_smoothing.grad_Te_scale must be > 0");
  }
  if (!(radiation.imc.net_e_source_smoothing.grad_rho_scale > 0.0)) {
    throw ValueError(
        "Radiation.imc.net_e_source_smoothing.grad_rho_scale must be > 0");
  }
  if (radiation.imc.conservative_smoother.passes < 0) {
    throw ValueError("Radiation.imc.conservative_smoother.passes must be >= 0");
  }
  if (!(radiation.imc.conservative_smoother.alpha > 0.0)) {
    throw ValueError("Radiation.imc.conservative_smoother.alpha must be > 0");
  }
  if (radiation.boundary.marshak_particles < 1) {
    tenryu::core::log_warning(
        "Radiation.boundary.marshak_particles < 1; clamping to 1");
    radiation.boundary.marshak_particles = 1;
  }
  if (!(radiation.ddmc.tau_ddmc >= 1.0)) {
    throw ValueError("Radiation.ddmc.tau_ddmc must be >= 1");
  }
  if (!(radiation.ddmc.tau_rw >= 0.0)) {
    throw ValueError("Radiation.ddmc.tau_rw must be >= 0");
  }
  if (radiation.ddmc.tau_ddmc_off >= 0.0 &&
      !(radiation.ddmc.tau_ddmc_off >= 0.5 &&
        radiation.ddmc.tau_ddmc_off <= radiation.ddmc.tau_ddmc)) {
    throw ValueError("Radiation.ddmc.tau_ddmc_off must satisfy "
                     "tau_ddmc_off < 0 or (0.5 <= tau_ddmc_off <= tau_ddmc)");
  }
  if (radiation.ddmc.omega_ddmc_off >= 0.0 &&
      !(radiation.ddmc.omega_ddmc_off >= 0.0 &&
        radiation.ddmc.omega_ddmc_off <= radiation.ddmc.omega_ddmc)) {
    throw ValueError("Radiation.ddmc.omega_ddmc_off must satisfy "
                     "omega_ddmc_off < 0 or (0 <= omega_ddmc_off <= omega_ddmc)");
  }
  if (radiation.ddmc.mode_hold < 0 || radiation.ddmc.mode_hold > 100) {
    throw ValueError("Radiation.ddmc.mode_hold must satisfy 0 <= mode_hold <= 100");
  }
  if (!(radiation.ddmc.rate_max > 0.0)) {
    throw ValueError("Radiation.ddmc.rate_max must be > 0");
  }
  if (!(radiation.diffusion.tau_on >= radiation.diffusion.tau_off &&
        radiation.diffusion.tau_off > 0.0)) {
    throw ValueError(
        "Radiation.diffusion must satisfy tau_on >= tau_off > 0");
  }
  if (!(radiation.diffusion.reduced_flux_on >= 0.0 &&
        radiation.diffusion.reduced_flux_on <= radiation.diffusion.reduced_flux_off &&
        radiation.diffusion.reduced_flux_off <= 1.0)) {
    throw ValueError(
        "Radiation.diffusion must satisfy 0 <= reduced_flux_on <= reduced_flux_off <= 1");
  }
  if (radiation.diffusion.mode_update_interval < 1) {
    throw ValueError("Radiation.diffusion.mode_update_interval must be >= 1");
  }
  if (radiation.diffusion.min_diffusion_island_cells < 1) {
    throw ValueError("Radiation.diffusion.min_diffusion_island_cells must be >= 1");
  }
  if (radiation.diffusion.imc_guard_cells < 1) {
    throw ValueError("Radiation.diffusion.imc_guard_cells must be >= 1");
  }
  if (radiation.diffusion.sts_max_stages < 0) {
    throw ValueError("Radiation.diffusion.sts_max_stages must be >= 0");
  }
  if (!(radiation.diffusion.sts_damping > 0.0 &&
        radiation.diffusion.sts_damping < 1.0)) {
    throw ValueError("Radiation.diffusion.sts_damping must be in (0, 1)");
  }
  if (!(radiation.diffusion.sts_subcycle_eta > 0.0 &&
        radiation.diffusion.sts_subcycle_eta <= 1.0)) {
    throw ValueError("Radiation.diffusion.sts_subcycle_eta must be in (0, 1]");
  }
  if (radiation.diffusion.interface_particles_per_face_group < 1) {
    throw ValueError(
        "Radiation.diffusion.interface_particles_per_face_group must be >= 1");
  }
  if (radiation.diffusion.exit_particles_per_cell_group < 1) {
    throw ValueError(
        "Radiation.diffusion.exit_particles_per_cell_group must be >= 1");
  }
  if (!(radiation.diffusion.lte_entry_energy_fraction_cap >= 0.0)) {
    throw ValueError(
        "Radiation.diffusion.lte_entry_energy_fraction_cap must be >= 0");
  }
  if (!(radiation.holo.coupling_tau >= 0.0)) {
    throw ValueError("Radiation.holo.coupling_tau must be >= 0");
  }
  if (radiation.holo.guard_cells < 0) {
    throw ValueError("Radiation.holo.guard_cells must be >= 0");
  }
  if (radiation.holo.blend_cells < 0) {
    throw ValueError("Radiation.holo.blend_cells must be >= 0");
  }
  if (radiation.holo.min_lo_cells < 0) {
    throw ValueError("Radiation.holo.min_lo_cells must be >= 0");
  }
  if (!(radiation.holo.q_min >= 0.0 &&
        radiation.holo.q_min <= radiation.holo.q_max &&
        radiation.holo.q_max <= 1.0)) {
    throw ValueError("Radiation.holo must satisfy 0 <= q_min <= q_max <= 1");
  }
  const bool holo_legacy_tau =
      radiation.holo.tau_on == 0.0 && radiation.holo.tau_off == 0.0;
  if (!holo_legacy_tau &&
      !(radiation.holo.tau_on >= radiation.holo.tau_off &&
        radiation.holo.tau_off > 0.0)) {
    throw ValueError(
        "Radiation.holo must satisfy tau_on >= tau_off > 0, or tau_on=tau_off=0");
  }
  if (!(radiation.holo.reduced_flux_on >= 0.0 &&
        radiation.holo.reduced_flux_on <= radiation.holo.reduced_flux_off &&
        radiation.holo.reduced_flux_off <= 1.0)) {
    throw ValueError(
        "Radiation.holo must satisfy 0 <= reduced_flux_on <= reduced_flux_off <= 1");
  }
  if (radiation.holo.update_interval < 1) {
    throw ValueError("Radiation.holo.update_interval must be >= 1");
  }
  if (radiation.holo.hold_on < 0) {
    throw ValueError("Radiation.holo.hold_on must be >= 0");
  }
  if (radiation.holo.min_dwell_steps < 0) {
    throw ValueError("Radiation.holo.min_dwell_steps must be >= 0");
  }
  if (radiation.holo.min_island_cells < 1) {
    throw ValueError("Radiation.holo.min_island_cells must be >= 1");
  }
  if (radiation.holo.core_margin_cells < 0) {
    throw ValueError("Radiation.holo.core_margin_cells must be >= 0");
  }
  if (radiation.holo.region != "shell") {
    throw ConfigError("Radiation.holo.region must be \"shell\" in v1");
  }
  if (radiation.holo.material_group != "shell") {
    throw ConfigError("Radiation.holo.material_group must be \"shell\" in v1");
  }
  if (radiation.holo.solver != "implicit_1d" &&
      radiation.holo.solver != "quasidiffusion_1d") {
    throw ConfigError(
        "Radiation.holo.solver must be \"implicit_1d\" or \"quasidiffusion_1d\"");
  }
  if (radiation.holo.closure != "diffusion") {
    throw ConfigError("Radiation.holo.closure must be \"diffusion\" in v1");
  }
  if (!(radiation.holo.closure_relax >= 0.0 &&
        radiation.holo.closure_relax <= 1.0)) {
    throw ValueError("Radiation.holo.closure_relax must be in [0, 1]");
  }
  if (radiation.holo.closure_smooth_passes < 0) {
    throw ValueError("Radiation.holo.closure_smooth_passes must be >= 0");
  }
  if (!(radiation.holo.closure_smooth_alpha >= 0.0 &&
        radiation.holo.closure_smooth_alpha <= 1.0)) {
    throw ValueError("Radiation.holo.closure_smooth_alpha must be in [0, 1]");
  }
  if (!(radiation.holo.consistency_alpha >= 0.0 &&
        radiation.holo.consistency_alpha <= 1.0)) {
    throw ValueError("Radiation.holo.consistency_alpha must be in [0, 1]");
  }
  if (radiation.holo.boundary_flux != "physical") {
    throw ConfigError(
        "Radiation.holo.boundary_flux must be \"physical\" in v1");
  }
  if (radiation.holo.sn_n_angles < 2 || (radiation.holo.sn_n_angles % 2) != 0) {
    throw ValueError("Radiation.holo.sn_n_angles must be an even integer >= 2");
  }
  if (radiation.holo.sn_material_coupling && !radiation.holo.sn_closure) {
    throw ValueError(
        "Radiation.holo.sn_material_coupling requires Radiation.holo.sn_closure=true");
  }
  if (radiation.holo.residual_particles_per_cell_group < 1) {
    throw ValueError(
        "Radiation.holo.residual_particles_per_cell_group must be >= 1");
  }
  if (radiation.holo.enabled) {
    tenryu::core::log_warning(
        "Radiation.holo.enabled=true: HOLO is experimental and not validated "
        "for production use. Use Radiation.imc.difference for production "
        "oscillation reduction.");
    if (radiation.imc.difference.enabled) {
      throw ValueError("Radiation.holo.enabled and Radiation.imc.difference.enabled "
                       "cannot both be true. HOLO+DF simultaneous operation is not "
                       "supported. Use difference formulation (DF) only for production.");
    }
  }
  if (radiation.holo.enabled && main.dimension == "2D_RZ" &&
      !radiation.holo.sn_material_coupling) {
    tenryu::core::log_warning(
        "Radiation.holo.enabled=True is only supported for Main.dimension=\"1D_SPH\" "
        "in v1; disabling HOLO for 2D_RZ");
    radiation.holo.enabled = false;
  }
  if (!radiation.compute_T_range_eV.empty() &&
      radiation.compute_T_range_eV.size() != 2U) {
    throw ConfigError("Radiation.compute_T_range_eV must be list[2] when specified");
  }
  if (!radiation.compute_T_range_eV.empty() &&
      !valid_temperature_range(radiation.compute_T_range_eV)) {
    throw ValueError(
        "Radiation.compute_T_range_eV must satisfy [Tmin>0, Tmax>Tmin]");
  }
  if (radiation.planck_fraction.method != "compute" &&
      radiation.planck_fraction.method != "tabulate") {
    throw ConfigError(
        "Radiation.groups.planck_fraction.method must be \"compute\" or \"tabulate\"");
  }
  if (radiation.planck_fraction.compute_N_T < 10) {
    throw ValueError("Radiation.groups.planck_fraction.compute_N_T must be >= 10");
  }
  if (!radiation.planck_fraction.compute_T_range_eV.empty() &&
      radiation.planck_fraction.compute_T_range_eV.size() != 2U) {
    throw ConfigError(
        "Radiation.groups.planck_fraction.compute_T_range_eV must be list[2] when specified");
  }
  if (!radiation.planck_fraction.compute_T_range_eV.empty() &&
      !valid_temperature_range(radiation.planck_fraction.compute_T_range_eV)) {
    throw ValueError(
        "Radiation.groups.planck_fraction.compute_T_range_eV must satisfy [Tmin>0, Tmax>Tmin]");
  }
  if (!radiation.group_bounds_eV.empty()) {
    const std::size_t expected_bounds_size =
        static_cast<std::size_t>(radiation.groups) + 1U;
    if (radiation.group_bounds_eV.size() != expected_bounds_size) {
      throw ConfigError("Radiation.group_bounds_eV size must be groups + 1");
    }
  }
  if (!radiation.group_bounds_eV.empty()) {
    for (std::size_t i = 0; i < radiation.group_bounds_eV.size(); ++i) {
      if (!(radiation.group_bounds_eV[i] >= 0.0)) {
        throw ConfigError("Radiation.group_bounds_eV must be >= 0");
      }
    }
    for (std::size_t i = 1; i < radiation.group_bounds_eV.size(); ++i) {
      if (!(radiation.group_bounds_eV[i] > radiation.group_bounds_eV[i - 1])) {
        throw ConfigError("Radiation.group_bounds_eV must be strictly increasing");
      }
    }
  }
  if (radiation.multigroup_diffusion.flux_limiter != "levermore_pomraning" &&
      radiation.multigroup_diffusion.flux_limiter != "larsen" &&
      radiation.multigroup_diffusion.flux_limiter != "none") {
    throw ConfigError("Radiation.multigroup_diffusion.flux_limiter must be one of "
                      "{\"levermore_pomraning\", \"larsen\", \"none\"}");
  }
  if (radiation.multigroup_diffusion.max_outer_iterations < 1) {
    throw ValueError(
        "Radiation.multigroup_diffusion.max_outer_iterations must be >= 1");
  }
  if (!(radiation.multigroup_diffusion.outer_tol > 0.0)) {
    throw ValueError("Radiation.multigroup_diffusion.outer_tol must be > 0");
  }
  if (radiation.multigroup_diffusion.state_supply_boundary_policy !=
          "local_D_current" &&
      radiation.multigroup_diffusion.state_supply_boundary_policy !=
          "harmonic_ghost_D_test" &&
      radiation.multigroup_diffusion.state_supply_boundary_policy !=
          "radial_mean_D_test") {
    throw ConfigError(
        "Radiation.multigroup_diffusion.state_supply_boundary_policy must be one of "
        "{\"local_D_current\", \"harmonic_ghost_D_test\", \"radial_mean_D_test\"}");
  }
  if (!(radiation.multigroup_diffusion.cg_inner_tol > 0.0)) {
    throw ValueError("Radiation.multigroup_diffusion.cg_inner_tol must be > 0");
  }
  if (radiation.multigroup_diffusion.cg_tol_norm != "r0" &&
      radiation.multigroup_diffusion.cg_tol_norm != "rhs") {
    throw ConfigError(
        "Radiation.multigroup_diffusion.cg_tol_norm must be one of "
        "{\"r0\", \"rhs\"}");
  }
  if (radiation.multigroup_diffusion.outer_accel != "none" &&
      radiation.multigroup_diffusion.outer_accel != "anderson") {
    throw ConfigError(
        "Radiation.multigroup_diffusion.outer_accel must be one of "
        "{\"none\", \"anderson\"}");
  }
  if (radiation.multigroup_diffusion.anderson_m < 1 ||
      radiation.multigroup_diffusion.anderson_m > 4) {
    throw ValueError(
        "Radiation.multigroup_diffusion.anderson_m must be in [1, 4]");
  }
  if (!(radiation.multigroup_diffusion.anderson_beta > 0.0) ||
      radiation.multigroup_diffusion.anderson_beta > 1.0) {
    throw ValueError(
        "Radiation.multigroup_diffusion.anderson_beta must be in (0, 1]");
  }
  if (radiation.multigroup_diffusion.cap_exit_policy != "warn" &&
      radiation.multigroup_diffusion.cap_exit_policy != "fail") {
    throw ConfigError(
        "Radiation.multigroup_diffusion.cap_exit_policy must be one of "
        "{\"warn\", \"fail\"}");
  }
  tenryu::core::validate_multigroup_diffusion_config(config);
  if (radiation.multigroup_diffusion.linear_solver_1d != "cusparse_tridiag") {
    throw ConfigError(
        "Radiation.multigroup_diffusion.linear_solver_1d must be \"cusparse_tridiag\"");
  }
  if (radiation.multigroup_diffusion.linear_solver_2d_requested.empty()) {
    radiation.multigroup_diffusion.linear_solver_2d_requested =
        radiation.multigroup_diffusion.linear_solver_2d;
  }
  radiation.multigroup_diffusion.linear_solver_2d_resolved =
      radiation.multigroup_diffusion.linear_solver_2d;
  if (radiation.multigroup_diffusion.linear_solver_2d != "auto" &&
      radiation.multigroup_diffusion.linear_solver_2d != "amgx_cg" &&
      radiation.multigroup_diffusion.linear_solver_2d != "jacobi" &&
      radiation.multigroup_diffusion.linear_solver_2d != "cusparse_cg_jacobi" &&
      radiation.multigroup_diffusion.linear_solver_2d != "cusparse_cg_zline" &&
      radiation.multigroup_diffusion.linear_solver_2d != "cusparse_cg_rgmg") {
    throw ConfigError(
        "Radiation.multigroup_diffusion.linear_solver_2d must be \"auto\", "
        "\"amgx_cg\", \"jacobi\", \"cusparse_cg_jacobi\", "
        "\"cusparse_cg_zline\", or \"cusparse_cg_rgmg\"");
  }
  if (main.dimension == "2D_RZ" &&
      radiation.enabled &&
      radiation.mode == RadiationMode::MultigroupDiffusion &&
      radiation.multigroup_diffusion.linear_solver_2d == "auto") {
    const int nr = config.mesh.nr;
    const int nz = config.mesh.nz;
    const bool nr_pow2 = nr > 0 && (nr & (nr - 1)) == 0;
    std::string resolved_value = "cusparse_cg_jacobi";
    if (nr_pow2 && nz >= 3) {
      resolved_value = "cusparse_cg_rgmg";
    } else if (nz >= 3) {
      resolved_value = "cusparse_cg_zline";
    }
    tenryu::core::log_info(
        "linear_solver_2d=auto resolved to " + resolved_value +
        " (nr=" + std::to_string(nr) + ", nz=" + std::to_string(nz) + ")");
    radiation.multigroup_diffusion.linear_solver_2d = resolved_value;
    radiation.multigroup_diffusion.linear_solver_2d_resolved = resolved_value;
  }
#if !(TENRYU_HAVE_AMGX && TENRYU_ENABLE_AMGX)
  if (main.dimension == "2D_RZ" &&
      radiation.enabled &&
      radiation.mode == RadiationMode::MultigroupDiffusion &&
      radiation.multigroup_diffusion.linear_solver_2d == "amgx_cg") {
    if (radiation.multigroup_diffusion.linear_solver_2d_explicit) {
      throw ConfigError(
          "linear_solver_2d=amgx_cg requested but AmgX is not linked in this "
          "build; choose cusparse_cg_zline / cusparse_cg_rgmg / jacobi "
          "explicitly");
    }
    core::log_warning("linear_solver_2d defaulted to jacobi (AmgX not linked)");
    radiation.multigroup_diffusion.linear_solver_2d = "jacobi";
    radiation.multigroup_diffusion.linear_solver_2d_resolved = "jacobi";
  }
#endif
  if (!(radiation.multigroup_diffusion.opacity_floor >= 0.0)) {
    throw ValueError("Radiation.multigroup_diffusion.opacity_floor must be >= 0");
  }
  if (!(radiation.multigroup_diffusion.opacity_cap >
        radiation.multigroup_diffusion.opacity_floor)) {
    throw ValueError(
        "Radiation.multigroup_diffusion.opacity_cap must be > opacity_floor");
  }
  if (radiation.multigroup_diffusion.boundary.inner_r != "reflect") {
    throw ConfigError(
        "Radiation.multigroup_diffusion.boundary.inner_r must be \"reflect\" in Cut-1a");
  }
  if (radiation.multigroup_diffusion.boundary.outer_r != "vacuum" &&
      radiation.multigroup_diffusion.boundary.outer_r != "reflect" &&
      !(main.dimension == "1D_SPH" &&
        radiation.multigroup_diffusion.boundary.outer_r == "marshak")) {
    throw ConfigError(
        "Radiation.multigroup_diffusion.boundary.outer_r must be \"vacuum\" or "
        "\"reflect\" (or \"marshak\" for Main.dimension=\"1D_SPH\").");
  }
  const auto supported_fld_z_boundary = [](const std::string& value) {
    return value == "vacuum" || value == "reflect" || value == "marshak" ||
           value == "state_supply";
  };
  if (!supported_fld_z_boundary(radiation.multigroup_diffusion.z_boundary)) {
    throw ConfigError(
        "Radiation.multigroup_diffusion.z_boundary must be \"vacuum\", \"reflect\", \"marshak\", or \"state_supply\"");
  }
  if (!supported_fld_z_boundary(radiation.multigroup_diffusion.boundary.z)) {
    throw ConfigError(
        "Radiation.multigroup_diffusion.boundary.z must be \"vacuum\", \"reflect\", \"marshak\", or \"state_supply\"");
  }
  if (!supported_fld_z_boundary(
          radiation.multigroup_diffusion.boundary.z_bottom)) {
    throw ConfigError(
        "Radiation.multigroup_diffusion.boundary.z_bottom must be \"vacuum\", \"reflect\", \"marshak\", or \"state_supply\"");
  }
  if (!supported_fld_z_boundary(radiation.multigroup_diffusion.boundary.z_top)) {
    throw ConfigError(
        "Radiation.multigroup_diffusion.boundary.z_top must be \"vacuum\", \"reflect\", \"marshak\", or \"state_supply\"");
  }
  const bool fld_has_marshak =
      radiation.multigroup_diffusion.boundary.z_bottom == "marshak" ||
      radiation.multigroup_diffusion.boundary.z_top == "marshak";
  if (!(radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s >= 0.0)) {
    throw ValueError(
        "Radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s must be >= 0");
  }
  if (radiation.multigroup_diffusion.marshak.flux_pulse_duration_s < 0.0 &&
      radiation.multigroup_diffusion.marshak.flux_pulse_duration_s != -1.0) {
    throw ValueError(
        "Radiation.multigroup_diffusion.marshak.flux_pulse_duration_s must be >= 0 or -1");
  }
  if (fld_has_marshak) {
    // Indirect-drive Tr(t) route (docs/design/2d_tr_drive_port_spec.md §5):
    // exactly one of {a Radiation.boundary marshak_Tr source, the constant
    // marshak.flux_erg_per_cm2_s > 0} drives the marshak z faces (the 1D
    // 819368bc pattern). The Tr route supplies per-group Planck weighting,
    // so the grey restriction stays on the flux route only.
    const auto fld_z_face_has_tr_table = [&](const char* key,
                                             const char* alias) {
      const auto& tr_map = radiation.boundary.marshak_Tr_map;
      return tr_map.count(key) > 0 || tr_map.count(alias) > 0;
    };
    const bool fld_bottom_marshak =
        radiation.multigroup_diffusion.boundary.z_bottom == "marshak";
    const bool fld_top_marshak =
        radiation.multigroup_diffusion.boundary.z_top == "marshak";
    const bool has_tr =
        radiation.boundary.marshak_Tr_eV > 0.0 ||
        radiation.boundary.marshak_Tr.detected ||
        (fld_bottom_marshak &&
         fld_z_face_has_tr_table("bottom_z", "z_bottom")) ||
        (fld_top_marshak && fld_z_face_has_tr_table("top_z", "z_top"));
    const bool has_flux =
        radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s > 0.0;
    if (has_tr == has_flux) {
      throw ConfigError(
          "Radiation.multigroup_diffusion marshak z boundary requires exactly"
          " one of a Radiation.boundary marshak_Tr source (marshak_Tr_eV > 0,"
          " a marshak_Tr time callable, or a marshak_Tr_map entry for the"
          " marshak face) or marshak.flux_erg_per_cm2_s > 0 (grey"
          " constant-flux drive)");
    }
    if (!has_tr && radiation.groups != 1) {
      throw ConfigError(
          "Radiation.multigroup_diffusion marshak boundary with the grey"
          " constant-flux drive is currently supported only for groups=1"
          " (the marshak_Tr route supplies per-group Planck weighting)");
    }
  }
  const bool fld_z_boundary_state_supply =
      radiation.multigroup_diffusion.z_boundary == "state_supply" ||
      radiation.multigroup_diffusion.boundary.z == "state_supply";
  const bool fld_bottom_state_supply =
      radiation.multigroup_diffusion.boundary.z_bottom == "state_supply";
  const bool fld_top_state_supply =
      radiation.multigroup_diffusion.boundary.z_top == "state_supply";
  if (fld_z_boundary_state_supply &&
      (!fld_bottom_state_supply || !fld_top_state_supply)) {
    throw ConfigError(
        "Radiation.multigroup_diffusion z_boundary='state_supply' requires both boundary.z_bottom and boundary.z_top to remain state_supply");
  }
  if (fld_bottom_state_supply || fld_top_state_supply) {
    if (main.dim != 2) {
      throw ConfigError(
          "Radiation.multigroup_diffusion state_supply boundary is supported only in 2D_RZ");
    }
    if (radiation.groups != 1) {
      throw ConfigError(
          "Radiation.multigroup_diffusion state_supply boundary is currently supported only for groups=1");
    }
    if (fld_bottom_state_supply &&
        !numerics.hydro.boundary_2d.z_bottom_cfg.is_state_supply()) {
      throw ConfigError(
          "Radiation.multigroup_diffusion boundary.z_bottom='state_supply' requires Numerics.hydro.boundary_2d.z_bottom.type='state_supply'");
    }
    if (fld_top_state_supply &&
        !numerics.hydro.boundary_2d.z_top_cfg.is_state_supply()) {
      throw ConfigError(
          "Radiation.multigroup_diffusion boundary.z_top='state_supply' requires Numerics.hydro.boundary_2d.z_top.type='state_supply'");
    }
    if ((fld_bottom_state_supply &&
         std::isfinite(
             numerics.hydro.boundary_2d.z_bottom_cfg.drive_t_end_s)) ||
        (fld_top_state_supply &&
         std::isfinite(numerics.hydro.boundary_2d.z_top_cfg.drive_t_end_s))) {
      throw ConfigError(
          "state_supply drive window is not supported with an FLD state_supply z-boundary in this version");
    }
  }
  const auto supported_sn_angles = [](const int n) {
    return n == 2 || n == 4 || n == 8 || n == 16 || n == 32;
  };
  // W-G3: 1D cylindrical S_N uses the product-quadrature counts 2*L*L
  // (8, 18, 32, 50, ...); the dedicated validation with the actionable
  // message lives in the W-G geometry block later in this function, so the
  // GL/level-symmetric whitelist must not fire first for that combination.
  const bool sn_cylindrical_product_counts =
      config.mesh.geometry_1d == "cylindrical" && radiation.enabled &&
      radiation.mode == RadiationMode::SnTransport;
  if (!sn_cylindrical_product_counts &&
      !supported_sn_angles(radiation.sn_transport.n_angles)) {
    throw ConfigError(
        "Radiation.sn_transport.n_angles must be one of {2, 4, 8, 16, 32}");
  }
  // wk finding (a): the FLD solver honors ONLY
  // Radiation.multigroup_diffusion.boundary; the top-level
  // Radiation.boundary face settings are silently ignored in FLD mode
  // (the GXII deck hides this by keeping both consistent). Reject
  // non-default top-level faces under FLD so decks cannot diverge
  // silently. The top-level marshak_Tr_eV/marshak_particles VALUE
  // channels remain legal (consumed by the marshak machinery).
  if (radiation.enabled &&
      radiation.mode == RadiationMode::MultigroupDiffusion) {
    const auto& b = radiation.boundary;
    if (b.type != "vacuum" || b.inner_r != "reflect" ||
        b.outer_r != "vacuum" || b.bottom_z != "vacuum" ||
        b.top_z != "vacuum") {
      throw ConfigError(
          "Radiation.mode=\"multigroup_diffusion\" ignores the top-level"
          " Radiation.boundary face settings; set"
          " Radiation.multigroup_diffusion.boundary instead");
    }
  }
  // wk finding (b): a Z=0 (neutral) material under the deterministic
  // radiation matter coupling explodes the Fleck solve (Cv_e-driven,
  // E_total blowup measured x1e23). Reject at namelist time.
  if (radiation.enabled &&
      (radiation.mode == RadiationMode::MultigroupDiffusion ||
       radiation.mode == RadiationMode::SnTransport)) {
    for (const auto& mat : materials.materials) {
      if (mat.is_void) {
        // Void materials are excluded from the radiation solve by the
        // cell_is_void masks; their normalized Z does not enter the
        // Fleck coupling.
        continue;
      }
      if (!(mat.Z > 0.0)) {
        throw ConfigError(
            "Radiation with mode multigroup_diffusion/sn_transport"
            " requires Z > 0 for every material (Fleck matter coupling"
            " diverges for neutral matter); material \"" + mat.name +
            "\" has Z <= 0 — disable radiation or set a positive Z");
      }
    }
  }
  const auto supported_sn_quadrature = [](const std::string& q) {
    return q == "level_symmetric_8" || q == "level_symmetric_16" ||
           q == "level_symmetric_32" || q == "product";
  };
  if (!supported_sn_quadrature(radiation.sn_transport.angular_quadrature)) {
    throw ConfigError(
        "Radiation.sn_transport.angular_quadrature must be one of "
        "{\"level_symmetric_8\", \"level_symmetric_16\", \"level_symmetric_32\", "
        "\"product\"}");
  }
  if (radiation.sn_transport.spatial_scheme != "diamond_difference" &&
      radiation.sn_transport.spatial_scheme != "linear_characteristic") {
    throw ConfigError(
        "Radiation.sn_transport.spatial_scheme must be \"diamond_difference\" "
        "or \"linear_characteristic\"");
  }
  if (radiation.sn_transport.max_outer_iterations < 1) {
    throw ValueError(
        "Radiation.sn_transport.max_outer_iterations must be >= 1");
  }
  if (radiation.sn_transport.max_inner_iterations < 1) {
    throw ValueError(
        "Radiation.sn_transport.max_inner_iterations must be >= 1");
  }
  if (!(radiation.sn_transport.outer_tol > 0.0)) {
    throw ValueError("Radiation.sn_transport.outer_tol must be > 0");
  }
  if (!(radiation.sn_transport.outer_tol_stagnation_factor > 0.0)) {
    throw ValueError(
        "Radiation.sn_transport.outer_tol_stagnation_factor must be > 0");
  }
  if (!(radiation.sn_transport.outer_tol_hydro_error_scale >= 0.0)) {
    throw ValueError(
        "Radiation.sn_transport.outer_tol_hydro_error_scale must be >= 0");
  }
  if (!(radiation.sn_transport.inner_tol > 0.0)) {
    throw ValueError("Radiation.sn_transport.inner_tol must be > 0");
  }
  if (radiation.sn_transport.inner_graph_unroll < 1) {
    throw ValueError("Radiation.sn_transport.inner_graph_unroll must be >= 1");
  }
  if (radiation.mode == RadiationMode::SnTransport &&
      radiation.sn_transport.n_angles == 8) {
    tenryu::core::log_warning(
        "Radiation.sn_transport.n_angles=8 selected; verify S8/S16 angular "
        "error is acceptable for production use");
  }
  if (radiation.sn_transport.diffusion_fallback_mode != "none" &&
      radiation.sn_transport.diffusion_fallback_mode != "per_group_hysteresis") {
    throw ConfigError(
        "Radiation.sn_transport.diffusion_fallback_mode must be \"none\" or "
        "\"per_group_hysteresis\"");
  }
  if (!(radiation.sn_transport.tau_diffusion_on >= 0.0)) {
    throw ValueError("Radiation.sn_transport.tau_diffusion_on must be >= 0");
  }
  if (!(radiation.sn_transport.tau_diffusion_off >= 0.0)) {
    throw ValueError("Radiation.sn_transport.tau_diffusion_off must be >= 0");
  }
  if (!(radiation.sn_transport.opacity_floor >= 0.0)) {
    throw ValueError("Radiation.sn_transport.opacity_floor must be >= 0");
  }
  if (!(radiation.sn_transport.opacity_cap >
        radiation.sn_transport.opacity_floor)) {
    throw ValueError(
        "Radiation.sn_transport.opacity_cap must be > opacity_floor");
  }
  if (radiation.sn_transport.boundary.inner_r != "reflect_parity" &&
      radiation.sn_transport.boundary.inner_r != "reflect") {
    throw ConfigError(
        "Radiation.sn_transport.boundary.inner_r must be \"reflect_parity\" or \"reflect\"; 2D_RZ SN hardwires the inner radial face to axis/parity reflect");
  }
  const auto supported_sn_outer_r = [&main](const std::string& value) {
    return value == "vacuum" || value == "reflect" ||
           (main.dimension == "1D_SPH" && value == "marshak");
  };
  if (!supported_sn_outer_r(radiation.sn_transport.boundary.outer_r)) {
    throw ConfigError(
        "Radiation.sn_transport.boundary.outer_r must be \"vacuum\" or "
        "\"reflect\" (or \"marshak\" for Main.dimension=\"1D_SPH\").");
  }
  if (radiation.sn_transport.boundary.outer_r == "reflect" &&
      main.dimension != "2D_RZ") {
    throw ConfigError(
        "Radiation.sn_transport.boundary.outer_r=\"reflect\" is supported only in 2D_RZ "
        "(1D_SPH outer face is hardcoded vacuum)");
  }
  const auto supported_sn_z_boundary = [](const std::string& value) {
    return value == "vacuum" || value == "reflect" || value == "marshak";
  };
  if (!supported_sn_z_boundary(radiation.sn_transport.z_boundary)) {
    throw ConfigError(
        "Radiation.sn_transport.z_boundary must be \"vacuum\", \"reflect\", or \"marshak\"");
  }
  if (!supported_sn_z_boundary(radiation.sn_transport.boundary.z)) {
    throw ConfigError(
        "Radiation.sn_transport.boundary.z must be \"vacuum\", \"reflect\", or \"marshak\"");
  }
  if (!supported_sn_z_boundary(radiation.sn_transport.boundary.z_bottom)) {
    throw ConfigError(
        "Radiation.sn_transport.boundary.z_bottom must be \"vacuum\", \"reflect\", or \"marshak\"");
  }
  if (!supported_sn_z_boundary(radiation.sn_transport.boundary.z_top)) {
    throw ConfigError(
        "Radiation.sn_transport.boundary.z_top must be \"vacuum\", \"reflect\", or \"marshak\"");
  }
  const bool sn_has_marshak =
      radiation.sn_transport.boundary.z_bottom == "marshak" ||
      radiation.sn_transport.boundary.z_top == "marshak";
  if (!(radiation.sn_transport.marshak.flux_erg_per_cm2_s >= 0.0)) {
    throw ValueError(
        "Radiation.sn_transport.marshak.flux_erg_per_cm2_s must be >= 0");
  }
  if (sn_has_marshak) {
    // Indirect-drive Tr(t) route (docs/design/2d_tr_drive_port_spec.md
    // §4-§5): exactly-one drive form; the 2D SN marshak injection is
    // structurally grey, so groups==1 binds on BOTH routes in v1.
    const auto sn_z_face_has_tr_table = [&](const char* key,
                                            const char* alias) {
      const auto& tr_map = radiation.boundary.marshak_Tr_map;
      return tr_map.count(key) > 0 || tr_map.count(alias) > 0;
    };
    const bool sn_bottom_marshak =
        radiation.sn_transport.boundary.z_bottom == "marshak";
    const bool sn_top_marshak =
        radiation.sn_transport.boundary.z_top == "marshak";
    const bool has_tr =
        radiation.boundary.marshak_Tr_eV > 0.0 ||
        radiation.boundary.marshak_Tr.detected ||
        (sn_bottom_marshak &&
         sn_z_face_has_tr_table("bottom_z", "z_bottom")) ||
        (sn_top_marshak && sn_z_face_has_tr_table("top_z", "z_top"));
    const bool has_flux =
        radiation.sn_transport.marshak.flux_erg_per_cm2_s > 0.0;
    if (has_tr == has_flux) {
      throw ConfigError(
          "Radiation.sn_transport marshak z boundary requires exactly one of"
          " a Radiation.boundary marshak_Tr source (marshak_Tr_eV > 0, a"
          " marshak_Tr time callable, or a marshak_Tr_map entry for the"
          " marshak face) or marshak.flux_erg_per_cm2_s > 0 (grey"
          " constant-flux drive)");
    }
    if (radiation.groups != 1) {
      throw ConfigError(
          "Radiation.sn_transport marshak boundary is currently supported"
          " only for groups=1 (the 2D SN marshak injection is structurally"
          " grey; spec docs/design/2d_tr_drive_port_spec.md §4.2)");
    }
    if (has_tr && sn_bottom_marshak && sn_top_marshak &&
        (sn_z_face_has_tr_table("bottom_z", "z_bottom") ||
         sn_z_face_has_tr_table("top_z", "z_top"))) {
      throw ConfigError(
          "2D_RZ SN dual-z marshak with per-face marshak_Tr_map tables is"
          " unsupported in v1 (one scalar drives both faces); use the scalar"
          " Radiation.boundary.marshak_Tr or marshak_Tr_eV");
    }
  }
  if (radiation.enabled) {
    const auto nonvoid_material_count = std::count_if(
        materials.materials.begin(), materials.materials.end(),
        [](const auto& mat) { return !mat.is_void; });
    if (nonvoid_material_count > 1 &&
        numerics.materials.per_material_conservation_enabled) {
      if (main.dimension != "2D_RZ") {
        throw ConfigError(
            "multi-material radiation is wired for 2D_RZ only (I4 G-1); 1D is not yet supported");
      }
    }
    if (nonvoid_material_count > 1 && main.dimension == "2D_RZ") {
      for (const auto& mat : materials.materials) {
        if (mat.is_void) {
          continue;
        }
        if (mat.opacity_model != "constant") {
          throw ConfigError(
              "multi-material radiation requires opacity.model=constant for all non-void materials "
              "(per-material blending of freq_dep/NLTE is future work; "
              "docs/design/i4_mm_rad_interface_spec.md)");
        }
      }
    }
  }
  if (main.dimension == "2D_RZ" &&
      numerics.hydro.plasma_viscosity.enabled &&
      numerics.materials.per_material_conservation_enabled) {
    throw ConfigError(
        "Numerics.hydro.plasma_viscosity is not supported with per-material "
        "conservation in 2D (v1)");
  }
  if (radiation.mode == RadiationMode::MultigroupDiffusion) {
    if (radiation.enabled &&
        main.dimension != "1D_SPH" && main.dimension != "2D_RZ") {
      throw ConfigError(
          "Radiation.mode=multigroup_diffusion requires Main.dimension=\"1D_SPH\" or \"2D_RZ\"");
    }
    const auto reject_conflict = [](const char* which) {
      throw ConfigError(std::string(
          "Radiation.mode=multigroup_diffusion requires imc/ddmc/holo/difference "
          "all disabled; got ") +
                        which + "=True");
    };
    if (radiation.imc.enabled) {
      reject_conflict("Radiation.imc.enabled");
    }
    if (radiation.ddmc.enabled) {
      reject_conflict("Radiation.ddmc.enabled");
    }
    if (radiation.holo.enabled) {
      reject_conflict("Radiation.holo.enabled");
    }
    if (radiation.imc.difference.enabled) {
      reject_conflict("Radiation.imc.difference.enabled");
    }
  }
  if (radiation.mode == RadiationMode::SnTransport) {
    if (radiation.enabled &&
        main.dimension != "1D_SPH" && main.dimension != "2D_RZ") {
      throw ConfigError(
          "Radiation.mode=sn_transport requires Main.dimension=\"1D_SPH\" or \"2D_RZ\"");
    }
    if (radiation.sn_transport.diffusion_fallback_mode != "none") {
      throw ConfigError(
          "Radiation.mode=sn_transport requires diffusion_fallback_mode=\"none\" in Cut-1b");
    }
    const auto reject_conflict = [](const char* which) {
      throw ConfigError(std::string(
          "Radiation.mode=sn_transport requires imc/ddmc/holo/difference "
          "all disabled; got ") +
                        which + "=True");
    };
    if (radiation.imc.enabled) {
      reject_conflict("Radiation.imc.enabled");
    }
    if (radiation.ddmc.enabled) {
      reject_conflict("Radiation.ddmc.enabled");
    }
    if (radiation.holo.enabled) {
      reject_conflict("Radiation.holo.enabled");
    }
    if (radiation.imc.difference.enabled) {
      reject_conflict("Radiation.imc.difference.enabled");
    }
  }
  if (!is_ddmc_leak_stencil(radiation.ddmc.leak_stencil)) {
    throw ConfigError(
        "Radiation.ddmc.leak_stencil must be \"4\" or \"9_kershaw\"");
  }
  if (!is_ddmc_interface_method(radiation.ddmc.interface_method)) {
    throw ConfigError("Radiation.ddmc.interface_method must be one of "
                      "{\"asymptotic_diffusion_limit\", \"marshak\", "
                      "\"cleveland_gentile\"}");
  }
  if (!is_ddmc_interface_exit_distribution(radiation.ddmc.interface_exit_distribution)) {
    throw ConfigError(
        "Radiation.ddmc.interface_exit_distribution must be \"cosine\" or \"half_isotropic\"");
  }
  if (!is_ddmc_face_opacity_temperature(radiation.ddmc.face_opacity_temperature)) {
    throw ConfigError(
        "Radiation.ddmc.face_opacity_temperature must be \"radiative_mean\"");
  }
  if (!radiation.enabled && radiation.ddmc.enabled) {
    tenryu::core::log_warning(
        "Radiation.enabled=False with ddmc.enabled=True; DDMC settings will be ignored");
  }
  if (radiation.ddmc.interface_method != "asymptotic_diffusion_limit") {
    tenryu::core::log_warning(
        "Radiation.ddmc.interface_method=\"" + radiation.ddmc.interface_method +
        "\" is parsed but not implemented in v1.0; using asymptotic_diffusion_limit");
  }
  if (radiation.imc.census_comb.enabled) {
    if (radiation.imc.census_comb.max_particles < 1) {
      throw ConfigError("Radiation.imc.census_comb.max_particles must be >= 1");
    }
    if (radiation.imc.census_comb.min_per_bin < 1) {
      throw ConfigError("Radiation.imc.census_comb.min_per_bin must be >= 1");
    }
    if (!(radiation.imc.census_comb.trigger_ratio > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.trigger_ratio must be > 0");
    }
    if (!(radiation.imc.census_comb.target_fraction > 0.0 &&
          radiation.imc.census_comb.target_fraction <= 1.0)) {
      throw ValueError("Radiation.imc.census_comb.target_fraction must be in (0, 1]");
    }
    if (!(radiation.imc.census_comb.mode_weight_imc > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.mode_weight_imc must be > 0");
    }
    if (!(radiation.imc.census_comb.mode_weight_ddmc > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.mode_weight_ddmc must be > 0");
    }
    if (!(radiation.imc.census_comb.adaptive_util_start >= 0.0 &&
          radiation.imc.census_comb.adaptive_util_start <
              radiation.imc.census_comb.adaptive_util_end &&
          radiation.imc.census_comb.adaptive_util_end <= 1.0)) {
      throw ValueError(
          "Radiation.imc.census_comb adaptive util must satisfy 0 <= start < end <= 1");
    }
    if (!(radiation.imc.census_comb.trigger_ratio >=
          radiation.imc.census_comb.trigger_ratio_floor)) {
      throw ValueError("Radiation.imc.census_comb.trigger_ratio must be >= trigger_ratio_floor");
    }
    if (!(radiation.imc.census_comb.trigger_ratio_floor >=
          radiation.imc.census_comb.target_fraction +
              radiation.imc.census_comb.trigger_hysteresis)) {
      throw ValueError(
          "Radiation.imc.census_comb.trigger_ratio_floor must be >= "
          "target_fraction + trigger_hysteresis");
    }
    if (!(radiation.imc.census_comb.trigger_hysteresis >= 0.0)) {
      throw ValueError("Radiation.imc.census_comb.trigger_hysteresis must be >= 0");
    }
    if (!(radiation.imc.census_comb.ess_min_tier0 > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.ess_min_tier0 must be > 0");
    }
    if (!(radiation.imc.census_comb.ess_min_tier1 > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.ess_min_tier1 must be > 0");
    }
    if (!(radiation.imc.census_comb.max_split_factor >= 1)) {
      throw ConfigError("Radiation.imc.census_comb.max_split_factor must be >= 1");
    }
  }

  if (laser.mode.empty()) {
    laser.mode = (main.dimension == "2D_RZ") ? "raytrace_3d" : "raytrace_2d";
  }
  laser.spherical_average = false;
  if (laser.mode == "spherical_average") {
    laser.spherical_average = true;
    laser.mode = "raytrace_2d";
  }
  if (laser.mode != "raytrace_2d" && laser.mode != "raytrace_3d" &&
      laser.mode != "radial_absorption_1d") {
    throw ConfigError("Laser mode '" + laser.mode + "' is unsupported");
  }
  if (main.dimension == "2D_RZ" && laser.mode == "radial_absorption_1d") {
    throw ConfigError(
        "Laser.mode=\"radial_absorption_1d\" requires Main.dimension=\"1D_SPH\"");
  }
  if (!laser_rays_per_beam_explicit) {
    laser.rays_per_beam = (main.dimension == "2D_RZ") ? 128 : 1000;
  }
  if (laser.rays_per_beam < 10) {
    throw ValueError("Laser.rays_per_beam must be >= 10");
  }
  if (laser.ray_output_count < 0) {
    throw ValueError("Laser.ray_output_count must be >= 0");
  }
  if (laser.enabled && laser.ray_output_count > laser.rays_per_beam) {
    throw ValueError(
        "Laser.ray_output_count must be <= Laser.rays_per_beam when Laser.enabled=True");
  }
  if (laser.ray_output_trajectory && laser.ray_output_count <= 0) {
    throw ValueError(
        "Laser.ray_output_trajectory requires Laser.ray_output_count > 0");
  }
  if (laser.ray_output_max_steps < 1 || laser.ray_output_max_steps > 100000) {
    throw ValueError("Laser.ray_output_max_steps must be in [1, 100000]");
  }
  if (!(laser.absorption.eps_n > 0.0 && laser.absorption.eps_n < 1.0)) {
    throw ValueError("Laser.absorption.eps_n must be in (0, 1)");
  }
  if (!(laser.absorption.coulomb_log_floor > 0.0)) {
    throw ValueError("Laser.absorption.coulomb_log_floor must be > 0");
  }
  if (!(laser.lasermesh.nr >= 4 && laser.lasermesh.nz >= 4)) {
    throw ValueError("Laser.lasermesh.nr/nz must be >= 4");
  }
  if (!(laser.lasermesh.nr_max >= 4)) {
    throw ValueError("Laser.lasermesh.nr_max must be >= 4");
  }
  if (!(laser.lasermesh.r_max_factor > 0.0 && laser.lasermesh.z_span_factor > 0.0)) {
    throw ValueError("Laser.lasermesh.r_max_factor/z_span_factor must be > 0");
  }
  if (!(laser.lasermesh.min_ratio > 0.0 && laser.lasermesh.min_ratio <= 1.0)) {
    throw ValueError("Laser.lasermesh.min_ratio must be in (0, 1]");
  }
  if (!(laser.lasermesh.mesh_factor > 0.0)) {
    throw ValueError("Laser.lasermesh.mesh_factor must be > 0");
  }
  if (!(laser.lasermesh.rmax_n_hat_threshold > 0.0 &&
        laser.lasermesh.rmax_n_hat_threshold < 1.0)) {
    throw ValueError("Laser.lasermesh.rmax_n_hat_threshold must be in (0, 1)");
  }
  if (!(laser.raytrace.cfl_ray > 0.0 && laser.raytrace.cfl_ray <= 1.0)) {
    throw ValueError("Laser.raytrace.cfl_ray must be in (0, 1]");
  }
  if (!(laser.raytrace.intensity_cutoff >= 0.0)) {
    throw ValueError("Laser.raytrace.intensity_cutoff must be >= 0");
  }
  if (!(laser.raytrace.eps_crit > 0.0 && laser.raytrace.eps_crit < 1.0)) {
    throw ValueError("Laser.raytrace.eps_crit must be in (0, 1)");
  }
  if (laser.raytrace.max_steps < 1) {
    throw ValueError("Laser.raytrace.max_steps must be >= 1");
  }
  if (laser.raytrace.max_steps > 100000) {
    throw ValueError("Laser.raytrace.max_steps must be <= 100000");
  }
  if (laser.raytrace.integrator != "leapfrog") {
    throw ConfigError("Laser.raytrace.integrator must be \"leapfrog\" in v1.0");
  }
  if (!std::isfinite(laser.raytrace.test_kappa)) {
    throw ValueError("Laser.raytrace.test_kappa must be finite");
  }
  if (!(laser.raytrace.ds_adapt_g_target > 0.0)) {
    throw ValueError("Laser.raytrace.ds_adapt_g_target must be > 0");
  }
  if (!(laser.raytrace.ds_adapt_tau_target > 0.0)) {
    throw ValueError("Laser.raytrace.ds_adapt_tau_target must be > 0");
  }
  if (!(laser.raytrace.ds_adapt_max_factor >= 1.0)) {
    throw ValueError("Laser.raytrace.ds_adapt_max_factor must be >= 1");
  }
  if (!(laser.raytrace_skip >= 0.0)) {
    throw ValueError("Laser.raytrace_skip must be >= 0");
  }
  if (!(laser.raytrace_skip_config.threshold >= 0.0)) {
    throw ValueError("Laser.raytrace_skip_config.threshold must be >= 0");
  }
  if (laser.raytrace_skip_config.max_consecutive < 1) {
    throw ValueError("Laser.raytrace_skip_config.max_consecutive must be >= 1");
  }
  if (laser.raytrace_skip_config.norm != "max_relative" &&
      laser.raytrace_skip_config.norm != "l2_relative") {
    throw ConfigError("Laser.raytrace_skip_config.norm must be \"max_relative\" or "
                      "\"l2_relative\"");
  }
  if (!(laser.raytrace_skip_config.crit_guard >= 0.0 &&
        laser.raytrace_skip_config.crit_guard < 1.0)) {
    throw ValueError("Laser.raytrace_skip_config.crit_guard must be in [0, 1)");
  }
  if (!(laser.deposit.conservation_tol > 0.0)) {
    throw ValueError("Laser.deposit.conservation_tol must be > 0");
  }
  if (laser.deposit.deposit_smooth_passes < 0) {
    throw ValueError("Laser.deposit.deposit_smooth_passes must be >= 0");
  }
  if (!(laser.deposit.deposit_smooth_alpha >= 0.0 &&
        laser.deposit.deposit_smooth_alpha <= 0.5)) {
    throw ValueError("Laser.deposit.deposit_smooth_alpha must be in [0, 0.5]");
  }
  if (laser.ib.zeff_model == "auto") {
    if (laser.ib.zeff_table.ndens > 0) {
      laser.ib.zeff_model = "table";
      tenryu::core::log_info(
          "Laser.ib.zeff_model=auto -> table (TMAT ionization fractions detected)");
    } else {
      laser.ib.zeff_model = "off";
    }
  }
  if (laser.ib.langdon_model == "auto") {
    std::string off_reason;
    if (!laser.enabled) {
      off_reason = "laser disabled";
    } else if (main.dimension != "1D_SPH") {
      off_reason = "not 1D_SPH";
    } else if (laser.mode == "radial_absorption_1d") {
      off_reason = "radial_absorption_1d path (Langdon not applied there)";
    } else {
      const std::string profile_error =
          langdon_profile_compatibility_error(laser);
      if (!profile_error.empty()) {
        off_reason = "beam profiles not vacuum-map compatible (" +
                     profile_error + ")";
      } else if (laser.beams.empty()) {
        off_reason = "no beams";
      }
    }
    if (off_reason.empty()) {
      laser.ib.langdon_model = "legacy_vacuum_map";
      laser.ib.langdon_auto_resolved = true;
      tenryu::core::log_info(
          "Laser.ib.langdon_model=auto -> legacy_vacuum_map (1D raytrace default-on)");
    } else {
      laser.ib.langdon_model = "off";
      tenryu::core::log_info(
          "Laser.ib.langdon_model=auto -> off (" + off_reason + ")");
    }
  }
  if (laser.ib.langdon_model == "legacy_vacuum_map" &&
      !laser.ib.langdon_auto_resolved) {
    if (laser.enabled && main.dimension != "1D_SPH") {
      throw ConfigError(
          "Laser.ib.langdon_model=legacy_vacuum_map requires 1D_SPH");
    }
    if (laser.enabled && laser.mode == "radial_absorption_1d") {
      throw ConfigError(
          "Laser.ib.langdon_model=legacy_vacuum_map is not supported for "
          "radial_absorption_1d");
    }
    const std::string profile_error =
        langdon_profile_compatibility_error(laser);
    if (!profile_error.empty()) {
      throw ConfigError(
          "Laser.ib.langdon_model=legacy_vacuum_map beam profiles are not "
          "vacuum-map compatible (" + profile_error + ")");
    }
  }
  if (laser.ib.zeff_model == "table") {
    if (laser.ib.zeff_table.ndens <= 0) {
      throw ConfigError(
          "Laser.ib.zeff_model=table requires a tmat material with an "
          "/ionization block");
    }
    if (tmat_ionization_count != 1) {
      throw ConfigError(
          "Laser.ib.zeff_model=table requires exactly one tmat material with "
          "an /ionization block");
    }
  }
  if (materials.zmoments.ndens > 0) {
    if (tmat_ionization_count != 1 ||
        materials.zmoments.provider_material < 0) {
      throw ConfigError(
          "TMAT ionization moments require exactly one tmat material with "
          "an /ionization block");
    }
    if (materials.materials.size() != 1U) {
      throw ConfigError(
          "TMAT ionization moments currently require a single-material problem");
    }
    if (main.dim != 1) {
      throw ConfigError("TMAT ionization moments are 1D_SPH-only in v1");
    }
  }
  const double critical_margin_min = 1.0 - laser.raytrace.eps_crit;
  if (std::isnan(laser.lasermesh.critical_margin)) {
    laser.lasermesh.critical_margin = critical_margin_min;
  }
  if (laser.lasermesh.critical_clip &&
      laser.lasermesh.critical_margin < critical_margin_min) {
    throw ConfigError("Laser.lasermesh.critical_margin must satisfy critical_margin >= "
                      "1 - Laser.raytrace.eps_crit");
  }
  if (laser.lasermesh.ghost_corona.enabled) {
    if (laser.lasermesh.ghost_corona.n_out < 1) {
      throw ValueError("Laser.lasermesh.ghost_corona.n_out must be >= 1");
    }
    if (!(laser.lasermesh.ghost_corona.ne_min_frac > 0.0)) {
      throw ValueError("Laser.lasermesh.ghost_corona.ne_min_frac must be > 0");
    }
    if (!(laser.lasermesh.ghost_corona.ne_max_frac >
          laser.lasermesh.ghost_corona.ne_min_frac)) {
      throw ValueError("Laser.lasermesh.ghost_corona.ne_max_frac must be > ne_min_frac");
    }
    if (!(laser.lasermesh.ghost_corona.Te_min_eV > 0.0)) {
      throw ValueError("Laser.lasermesh.ghost_corona.Te_min_eV must be > 0");
    }
    if (!(laser.lasermesh.ghost_corona.zbar_min > 0.0)) {
      throw ValueError("Laser.lasermesh.ghost_corona.zbar_min must be > 0");
    }
    if (!(laser.lasermesh.ghost_corona.zbar_max >=
          laser.lasermesh.ghost_corona.zbar_min)) {
      throw ValueError("Laser.lasermesh.ghost_corona.zbar_max must be >= zbar_min");
    }
    if (laser.lasermesh.ghost_corona.handoff_cells < 1) {
      throw ValueError("Laser.lasermesh.ghost_corona.handoff_cells must be >= 1");
    }
    if (!(laser.lasermesh.ghost_corona.handoff_decay > 0.0)) {
      throw ValueError("Laser.lasermesh.ghost_corona.handoff_decay must be > 0");
    }
    if (laser.lasermesh.ghost_corona.transition_enabled) {
      if (!(laser.lasermesh.ghost_corona.transition_resolved_nhat > 0.0)) {
        throw ValueError(
            "Laser.lasermesh.ghost_corona.transition_resolved_nhat must be > 0");
      }
      if (laser.lasermesh.ghost_corona.transition_resolved_cells < 1) {
        throw ValueError(
            "Laser.lasermesh.ghost_corona.transition_resolved_cells must be >= 1");
      }
      if (!(laser.lasermesh.ghost_corona.transition_density_exponent >= 0.0)) {
        throw ValueError(
            "Laser.lasermesh.ghost_corona.transition_density_exponent must be >= 0");
      }
    }
  }
  if (laser.enabled && laser.beams.empty()) {
    throw ConfigError("Laser.enabled=True requires at least one beam");
  }
  if (laser.enabled && main.dimension == "1D_SPH" &&
      laser.mode != "raytrace_2d" && laser.mode != "radial_absorption_1d") {
    throw ConfigError(
        "Laser.mode must be \"raytrace_2d\", \"spherical_average\", or "
        "\"radial_absorption_1d\" for Main.dimension=\"1D_SPH\"");
  }
  if (laser.enabled && main.dimension == "2D_RZ" && laser.mode != "raytrace_3d") {
    throw ConfigError("Laser.mode must be \"raytrace_3d\" for Main.dimension=\"2D_RZ\"");
  }

  // W-B: 1D_SPH FLD boundary/source contract (NUMERICS §6.7). Reject
  // silently-ignored or ambiguous settings at config time instead of letting
  // the runtime default them away.
  if (main.dimension == "1D_SPH" && radiation.enabled &&
      radiation.mode == RadiationMode::MultigroupDiffusion) {
    const auto& fldb = radiation.multigroup_diffusion.boundary;
    if (fldb.outer_r != "vacuum" && fldb.outer_r != "reflect" &&
        fldb.outer_r != "marshak") {
      throw ConfigError(
          "Radiation.multigroup_diffusion.boundary.outer_r must be \"vacuum\","
          " \"reflect\", or \"marshak\" for Main.dimension=\"1D_SPH\"");
    }
    if (fldb.inner_r != "reflect") {
      throw ConfigError(
          "Radiation.multigroup_diffusion.boundary.inner_r must be \"reflect\""
          " for Main.dimension=\"1D_SPH\" (r=0 is the spherical center)");
    }
    if (fldb.z != "vacuum" || fldb.z_bottom != "vacuum" ||
        fldb.z_top != "vacuum") {
      throw ConfigError(
          "Radiation.multigroup_diffusion.boundary.z/z_bottom/z_top are"
          " 2D_RZ-only and ignored for Main.dimension=\"1D_SPH\"; remove them");
    }
    if (fldb.outer_r == "marshak") {
      const bool has_tr = radiation.boundary.marshak_Tr_eV > 0.0 ||
                          radiation.boundary.marshak_Tr.detected;
      const bool has_flux =
          radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s > 0.0;
      if (has_tr == has_flux) {
        throw ConfigError(
            "1D_SPH FLD marshak outer boundary requires exactly one of"
            " Radiation.boundary.marshak_Tr_eV > 0 (blackbody drive) or a"
            " Radiation.boundary.marshak_Tr time callable or"
            " Radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s > 0"
            " (grey constant-flux drive)");
      }
      if (has_flux && radiation.groups != 1) {
        throw ConfigError(
            "Radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s"
            " requires groups=1 (grey) for 1D_SPH; use"
            " Radiation.boundary.marshak_Tr_eV for a multigroup blackbody"
            " drive");
      }
    }
    if (radiation.volume_source_rate > 0.0) {
      if (radiation.groups != 1) {
        throw ConfigError(
            "Radiation.volume_source_rate requires groups=1 (grey) for"
            " 1D_SPH multigroup_diffusion");
      }
      if (!(radiation.volume_source_x_max > 0.0)) {
        throw ConfigError(
            "Radiation.volume_source_x_max must be > 0 when"
            " Radiation.volume_source_rate > 0");
      }
    }
  }

  // W-G: non-spherical 1D geometry support boundaries.
  if (config.mesh.geometry_1d != "spherical") {
    if (main.dimension != "1D_SPH") {
      throw ConfigError(
          "Mesh.geometry_1d is 1D-only; remove it for Main.dimension=\"2D_RZ\"");
    }
    if (radiation.enabled && radiation.mode == RadiationMode::ImcDdmc) {
      throw ConfigError(
          "Mesh.geometry_1d != \"spherical\" is not supported for the legacy"
          " imc_ddmc radiation mode");
    }
    if (config.mesh.geometry_1d == "cylindrical" && radiation.enabled &&
        radiation.mode == RadiationMode::SnTransport) {
      // W-G3: 1D cylindrical S_N uses the product angular quadrature
      // (NUMERICS section 6.8.3): n_angles must be 2*L*L with integer L >= 2.
      const int n = radiation.sn_transport.n_angles;
      const int half = n / 2;
      int L = 0;
      while ((L + 1) * (L + 1) <= half) {
        ++L;
      }
      if (n < 8 || (n % 2) != 0 || L * L != half || L > 32) {
        throw ConfigError(
            "Mesh.geometry_1d=\"cylindrical\" with sn_transport requires "
            "Radiation.sn_transport.n_angles = 2*L*L with integer 2 <= L <= 32 "
            "(8, 18, 32, 50, 72, ...); got " + std::to_string(n));
      }
    }
    if (laser.enabled && laser.mode != "radial_absorption_1d") {
      throw ConfigError(
          "Mesh.geometry_1d != \"spherical\" requires"
          " Laser.mode=\"radial_absorption_1d\" (raytrace_2d's beam-local"
          " mesh is spherical-only)");
    }
    if (numerics.conduction.test_planar) {
      throw ConfigError(
          "Numerics.conduction.test_planar together with a non-spherical"
          " Mesh.geometry_1d is ambiguous; drop test_planar (geometry_1d"
          " drives the conduction geometry)");
    }
  }

  // W-B2: 1D_SPH SN outer-boundary contract (NUMERICS §6.8 1D BC).
  if (main.dimension == "1D_SPH" && radiation.enabled &&
      radiation.mode == RadiationMode::SnTransport) {
    const auto& snb = radiation.sn_transport.boundary;
    if (snb.outer_r != "vacuum" && snb.outer_r != "marshak") {
      throw ConfigError(
          "Radiation.sn_transport.boundary.outer_r must be \"vacuum\" or"
          " \"marshak\" for Main.dimension=\"1D_SPH\"");
    }
    if (snb.outer_r == "marshak") {
      if (radiation.sn_transport.spatial_scheme != "linear_characteristic") {
        throw ConfigError(
            "1D_SPH SN marshak outer boundary requires"
            " sn_transport.spatial_scheme=\"linear_characteristic\"");
      }
      const bool has_tr = radiation.boundary.marshak_Tr_eV > 0.0 ||
                          radiation.boundary.marshak_Tr.detected;
      const bool has_flux =
          radiation.sn_transport.marshak.flux_erg_per_cm2_s > 0.0;
      if (has_tr == has_flux) {
        throw ConfigError(
            "1D_SPH SN marshak outer boundary requires exactly one of"
            " Radiation.boundary.marshak_Tr_eV > 0 (blackbody drive) or a"
            " Radiation.boundary.marshak_Tr time callable or"
            " Radiation.sn_transport.marshak.flux_erg_per_cm2_s > 0"
            " (grey constant-flux drive)");
      }
      if (has_flux && radiation.groups != 1) {
        throw ConfigError(
            "Radiation.sn_transport.marshak.flux_erg_per_cm2_s requires"
            " groups=1 (grey) for 1D_SPH");
      }
    }
  }

  if (main.dimension == "1D_SPH" && radiation.enabled) {
    const auto n_nonvoid_materials = std::count_if(
        materials.materials.begin(), materials.materials.end(),
        [](const auto& mat) { return !mat.is_void; });
    if (n_nonvoid_materials > 1) {
      if (radiation.mode == RadiationMode::SnTransport) {
        throw ConfigError(
            "Radiation.mode=\"sn_transport\" does not support multiple "
            "non-void materials in 1D_SPH (per-cell scattering-opacity "
            "mixing is not implemented); use a single non-void material");
      }
      if (radiation.mode == RadiationMode::MultigroupDiffusion) {
        for (std::size_t material_index = 0;
             material_index < materials.materials.size();
             ++material_index) {
          const auto& mat = materials.materials[material_index];
          if (mat.is_void) {
            continue;
          }
          if (mat.opacity_model != "constant" && mat.opacity_model != "none" &&
              mat.opacity_model != "tmat") {
            throw ConfigError(
                "Materials.materials[" + std::to_string(material_index) +
                "].opacity.model=\"" + mat.opacity_model +
                "\": 1D_SPH multi-material decks with "
                "Radiation.mode=\"multigroup_diffusion\" support "
                "per-material constant opacities and tmat tables; use "
                "opacity model \"constant\", \"none\", or \"tmat\", "
                "or reduce to a single non-void material");
          }
        }
      }
    }
  }

  if (numerics.conduction.sts_total_stages_max < 0) {
    throw ConfigError(
        "Numerics.conduction.sts_total_stages_max must be >= 0 "
        "(0 disables the bound)");
  }
  if (numerics.conduction.solver != "sts" && numerics.conduction.solver != "implicit" &&
      numerics.conduction.solver != "hypre") {
    throw ConfigError(
        "Numerics.conduction.solver must be \"sts\", \"implicit\", or \"hypre\"");
  }
  if (numerics.conduction.sts_floor_limiter != "net" &&
      numerics.conduction.sts_floor_limiter != "donor") {
    throw ConfigError(
        "Numerics.conduction.sts_floor_limiter must be \"net\" or \"donor\"");
  }
  if (numerics.conduction.face_kappa_policy != "harmonic" &&
      numerics.conduction.face_kappa_policy != "kirchhoff_same_material") {
    throw ConfigError(
        "Numerics.conduction.face_kappa_policy must be \"harmonic\" or"
        " \"kirchhoff_same_material\"");
  }
  tenryu::core::validate_hydro_av_config(config);
  tenryu::core::validate_tri_fan_stage2_config(config);
  tenryu::core::validate_button_stage1_config(config);
  if (main.dimension == "1D_CYL") {
    // 1D_CYL v1 scope: pure hydro core only
    // (docs/design/b1_1d_cyl_mode_spec.md section 3).
    if (radiation.enabled) {
      throw ConfigError(
          "Radiation.enabled=True is not supported for "
          "Main.dimension=\"1D_CYL\" (1D_CYL v1: pure hydro core only)");
    }
    if (laser.enabled) {
      throw ConfigError(
          "Laser.enabled=True is not supported for "
          "Main.dimension=\"1D_CYL\" (1D_CYL v1: pure hydro core only)");
    }
    if (numerics.conduction.enabled) {
      throw ConfigError(
          "Numerics.conduction.enabled=True is not supported for "
          "Main.dimension=\"1D_CYL\" (1D_CYL v1: pure hydro core only)");
    }
    if (numerics.ale1d.enabled) {
      throw ConfigError(
          "Numerics.ale1d.enabled=True is not supported for "
          "Main.dimension=\"1D_CYL\" (1D_CYL v1: pure hydro core only)");
    }
  }
  if (numerics.hydro.compatible_energy && main.dimension != "1D_SPH") {
    throw ConfigError(
        "Numerics.hydro.compatible_energy=True is supported only in 1D_SPH");
  }
  if (config.mesh.grading.mapping == "exact_measure_v2" &&
      main.dimension == "2D_RZ") {
    throw ConfigError(
        "Mesh.grid.grading.mapping=\"exact_measure_v2\" is 1D-only (v1); "
        "2D graded directions keep the legacy estimated-radius mapping");
  }
  if (numerics.hydro.time_integrator == "midpoint_v2" &&
      main.dimension == "2D_RZ") {
    throw ConfigError(
        "Numerics.hydro.time_integrator=\"midpoint_v2\" is 1D-only (v1)");
  }
  if (!(numerics.hydro.axis_motion_floor_fraction >= 0.0 &&
        numerics.hydro.axis_motion_floor_fraction <= 1.0)) {
    throw ValueError("Numerics.hydro.axis_motion_floor_fraction must be in [0, 1]");
  }
  if (!(numerics.hydro.axis_margin_dt_floor_fraction >= 0.0 &&
        numerics.hydro.axis_margin_dt_floor_fraction <= 1.0)) {
    throw ValueError("Numerics.hydro.axis_margin_dt_floor_fraction must be in [0, 1]");
  }
  if (!(numerics.hydro.volume_rate_cfl_threshold > 0.0)) {
    throw ValueError("Numerics.hydro.volume_rate_cfl_threshold must be > 0");
  }
  if (!(numerics.hydro.av_qcap_over_p >= 0.0)) {
    throw ValueError("Numerics.hydro.av_qcap_over_p must be >= 0");
  }
  if (!(numerics.hydro.av_cfl_coefficient > 0.0)) {
    throw ValueError("Numerics.hydro.av_cfl_coefficient must be > 0");
  }
  if (!(numerics.hydro.tri_fan_center_cfl_safety > 0.0)) {
    throw ValueError("Numerics.hydro.tri_fan_center_cfl_safety must be > 0");
  }
  if (numerics.hydro.tri_fan_center_cfl_band_radial_index < 0) {
    throw ValueError(
        "Numerics.hydro.tri_fan_center_cfl_band_radial_index must be >= 0");
  }
  if (!(numerics.hydro.corner_j_predict_cfl_safety > 0.0 &&
        numerics.hydro.corner_j_predict_cfl_safety <= 1.0)) {
    throw ValueError(
        "Numerics.hydro.corner_j_predict_cfl_safety must be in (0, 1]");
  }
  if (!(numerics.hydro.corner_j_predict_floor_frac > 0.0 &&
        numerics.hydro.corner_j_predict_floor_frac <= 1.0)) {
    throw ValueError(
        "Numerics.hydro.corner_j_predict_floor_frac must be in (0, 1]");
  }
  if (!(numerics.hydro.corner_j_predict_max_shrink > 0.0 &&
        numerics.hydro.corner_j_predict_max_shrink < 1.0)) {
    throw ValueError(
        "Numerics.hydro.corner_j_predict_max_shrink must be in (0, 1)");
  }
  if (numerics.hydro.corner_j_predict_shell_rings < 0) {
    throw ValueError(
        "Numerics.hydro.corner_j_predict_shell_rings must be >= 0");
  }
  if (!(numerics.hydro.rz_geometric_cfl_etaV > 0.0 &&
        numerics.hydro.rz_geometric_cfl_etaV <= 1.0)) {
    throw ValueError("Numerics.hydro.rz_geometric_cfl_etaV must be in (0, 1]");
  }
  if (!(numerics.hydro.rz_geometric_cfl_r_floor >= 0.0)) {
    throw ValueError("Numerics.hydro.rz_geometric_cfl_r_floor must be >= 0");
  }
  if (!(numerics.hydro.rz_geometric_cfl_v_initial_floor >= 0.0 &&
        numerics.hydro.rz_geometric_cfl_v_initial_floor <= 1.0)) {
    throw ValueError("Numerics.hydro.rz_geometric_cfl_v_initial_floor must be in [0, 1]");
  }
  if (!(numerics.hydro.trial_volume_cfl_floor_fraction > 0.0 &&
        numerics.hydro.trial_volume_cfl_floor_fraction <= 1.0)) {
    throw ValueError("Numerics.hydro.trial_volume_cfl_floor_fraction must be in (0, 1]");
  }
  if (!(numerics.hydro.trial_volume_cfl_shrink_fraction > 0.0 &&
        numerics.hydro.trial_volume_cfl_shrink_fraction < 1.0)) {
    throw ValueError("Numerics.hydro.trial_volume_cfl_shrink_fraction must be in (0, 1)");
  }
  if (!(numerics.hydro.corner_jacobian_floor_eps >= 0.0 &&
        numerics.hydro.corner_jacobian_floor_eps < 1.0)) {
    throw ValueError("Numerics.hydro.corner_jacobian_floor_eps must be in [0, 1)");
  }
  if (!(numerics.hydro.corner_jacobian_ale_trigger_scale > 0.0 &&
        numerics.hydro.corner_jacobian_ale_trigger_scale <= 1.0)) {
    throw ValueError("Numerics.hydro.corner_jacobian_ale_trigger_scale must be in (0, 1]");
  }
  if (!(numerics.hydro.in_hydro_gauss_j_floor_rel > 0.0)) {
    throw ValueError("Numerics.hydro.in_hydro_gauss_j_floor_rel must be > 0");
  }
  if (!(numerics.hydro.in_hydro_rz_volume_floor_rel > 0.0)) {
    throw ValueError("Numerics.hydro.in_hydro_rz_volume_floor_rel must be > 0");
  }
  if (!(numerics.hydro.mesh_quality_dt_safety_alpha > 0.0 &&
        numerics.hydro.mesh_quality_dt_safety_alpha <= 1.0)) {
    throw ValueError(
        "Numerics.hydro.mesh_quality_dt_safety_alpha must be in (0, 1]");
  }
  if (!(numerics.hydro.mesh_quality_dt_corner_j_floor_rel > 0.0)) {
    throw ValueError("Numerics.hydro.mesh_quality_dt_corner_j_floor_rel must be > 0");
  }
  if (!(numerics.hydro.mesh_quality_dt_gauss_j_floor_rel > 0.0)) {
    throw ValueError("Numerics.hydro.mesh_quality_dt_gauss_j_floor_rel must be > 0");
  }
  if (!(numerics.hydro.mesh_quality_dt_rz_volume_floor_rel > 0.0)) {
    throw ValueError(
        "Numerics.hydro.mesh_quality_dt_rz_volume_floor_rel must be > 0");
  }
  if (numerics.hydro.axis_guard_band_cells < 0) {
    throw ValueError("Numerics.hydro.axis_guard_band_cells must be >= 0");
  }
  if (numerics.hydro.driver_full_step_retry_max_attempts < 0) {
    throw ValueError(
        "Numerics.hydro.driver_full_step_retry_max_attempts must be >= 0");
  }
  if (numerics.hydro.dispatcher_state_sensitive_repair_cap_per_step < 1) {
    throw ValueError(
        "Numerics.hydro.dispatcher_state_sensitive_repair_cap_per_step must be >= 1");
  }
  if (numerics.hydro.strategy_first_max_same_dt_attempts < 0) {
    throw ValueError(
        "Numerics.hydro.strategy_first_max_same_dt_attempts must be >= 0");
  }
  if (!(numerics.hydro.driver_retry_corner_balance_threshold > 0.0 &&
        numerics.hydro.driver_retry_corner_balance_threshold < 1.0)) {
    throw ValueError(
        "Numerics.hydro.driver_retry_corner_balance_threshold must be in (0, 1)");
  }
  if (numerics.hydro.geometric_retry_stagnation.same_cell_count_threshold < 1) {
    throw ValueError(
        "Numerics.hydro.geometric_retry_stagnation."
        "same_cell_count_threshold must be >= 1");
  }
  if (!(numerics.hydro.geometric_retry_stagnation.sigma_rel_tol > 0.0 &&
        numerics.hydro.geometric_retry_stagnation.sigma_rel_tol <= 1.0)) {
    throw ValueError(
        "Numerics.hydro.geometric_retry_stagnation.sigma_rel_tol must be in (0, 1]");
  }
  if (!(numerics.hydro.geometric_retry_stagnation.dt_drop_factor > 0.0 &&
        numerics.hydro.geometric_retry_stagnation.dt_drop_factor < 1.0)) {
    throw ValueError(
        "Numerics.hydro.geometric_retry_stagnation.dt_drop_factor must be in (0, 1)");
  }
  if (numerics.hydro.ee_odd_even_C > 0.0) {
    if (main.dimension != "1D_SPH") {
      throw ConfigError("Numerics.hydro.ee_odd_even_C > 0 is supported only in 1D_SPH");
    }
    if (!main.two_temperature) {
      throw ConfigError("Numerics.hydro.ee_odd_even_C > 0 requires 2T hydro");
    }
  }
  if (!(numerics.hydro.anti_hourglass_kappa > 0.0)) {
    throw ValueError("Numerics.hydro.anti_hourglass_kappa must be > 0");
  }
  if (numerics.hydro.subzonal_pressure_mode != "uniform_cell") {
    throw ConfigError(
        "Numerics.hydro.subzonal_pressure_mode must be \"uniform_cell\" "
        "in the current implementation");
  }
  if (numerics.hydro.subzonal_band_mode != "off" &&
      numerics.hydro.subzonal_band_mode != "bridge_feather") {
    throw ValueError(
        "Numerics.hydro.subzonal_band_mode must be one of "
        "{\"off\", \"bridge_feather\"}");
  }
  if (numerics.hydro.subzonal_band_feather_layers < 1) {
    throw ValueError(
        "Numerics.hydro.subzonal_band_feather_layers must be >= 1");
  }
  if (numerics.hydro.subzonal_mass_enabled && main.dimension != "2D_RZ") {
    throw ConfigError(
        "Numerics.hydro.subzonal_mass_enabled=true is supported only in 2D_RZ");
  }
  if (numerics.hydro.bbs_axis_policy_enabled && main.dimension != "2D_RZ") {
    throw ConfigError(
        "Numerics.hydro.bbs_axis_policy_enabled=true is supported only in 2D_RZ");
  }
  if (numerics.hydro.subzonal_mass_lagrangian_invariant_enabled &&
      main.dimension != "2D_RZ") {
    throw ConfigError(
        "Numerics.hydro.subzonal_mass_lagrangian_invariant_enabled=true is supported only in 2D_RZ");
  }
  if (!(numerics.hydro.hourglass.scale > 0.0)) {
    throw ValueError("Numerics.hydro.hourglass.scale must be > 0");
  }
  if (!(numerics.hydro.hourglass.activation_corner_j_ratio_threshold > 0.0 &&
        numerics.hydro.hourglass.activation_corner_j_ratio_threshold <= 1.0)) {
    throw ValueError(
        "Numerics.hydro.hourglass.activation_corner_j_ratio_threshold must be in (0, 1]");
  }
  if (!(numerics.hydro.hourglass.activation_hourglass_amplitude_threshold > 0.0 &&
        numerics.hydro.hourglass.activation_hourglass_amplitude_threshold <= 1.0)) {
    throw ValueError(
        "Numerics.hydro.hourglass.activation_hourglass_amplitude_threshold must be in (0, 1]");
  }
  if (numerics.hydro.hourglass.subzonal_pressure_model != "linearized") {
    throw ConfigError(
        "Numerics.hydro.hourglass.subzonal_pressure_model must be \"linearized\" "
        "in the current implementation");
  }
  if (!(numerics.hydro.hourglass.max_force_per_node_fraction > 0.0)) {
    throw ValueError("Numerics.hydro.hourglass.max_force_per_node_fraction must be > 0");
  }
  if (numerics.hydro.hourglass.enabled && main.dimension != "2D_RZ") {
    throw ConfigError(
        "Numerics.hydro.hourglass.enabled=true is supported only in 2D_RZ");
  }
  if (numerics.hydro.total_energy_remap_2d_rz &&
      main.dimension != "2D_RZ") {
    throw ConfigError(
        "Numerics.hydro.total_energy_remap_2d_rz is supported only in 2D_RZ");
  }
  if (numerics.hydro.total_energy_remap_2d_rz &&
      numerics.materials.per_material_conservation_enabled) {
    throw ConfigError(
        "Numerics.hydro.total_energy_remap_2d_rz does not support per-material hydro");
  }
  if (numerics.hydro.work_split_audit_2d_rz &&
      main.dimension != "2D_RZ") {
    throw ConfigError(
        "Numerics.hydro.work_split_audit_2d_rz is supported only in 2D_RZ");
  }
  if (numerics.hydro.work_split_audit_2d_rz &&
      numerics.materials.per_material_conservation_enabled) {
    throw ConfigError(
        "Numerics.hydro.work_split_audit_2d_rz does not support per-material hydro");
  }
  if (numerics.hydro.hllc_z_flux_2d_rz &&
      main.dimension != "2D_RZ") {
    throw ConfigError(
        "Numerics.hydro.hllc_z_flux_2d_rz is supported only in 2D_RZ");
  }
  if (numerics.hydro.hllc_z_flux_2d_rz &&
      !numerics.hydro.total_energy_remap_2d_rz) {
    throw ConfigError(
        "Numerics.hydro.hllc_z_flux_2d_rz requires "
        "Numerics.hydro.total_energy_remap_2d_rz=true");
  }
  if (numerics.hydro.hllc_z_flux_2d_rz &&
      numerics.materials.per_material_conservation_enabled) {
    throw ConfigError(
        "Numerics.hydro.hllc_z_flux_2d_rz does not support per-material hydro");
  }
  if (numerics.hydro.ion_art_heat_C > 0.0) {
    if (main.dimension != "1D_SPH") {
      throw ConfigError("Numerics.hydro.ion_art_heat_C > 0 is supported only in 1D_SPH");
    }
    if (!main.two_temperature) {
      throw ConfigError("Numerics.hydro.ion_art_heat_C > 0 requires 2T hydro");
    }
    if (numerics.hydro.av_type != "vnr") {
      throw ConfigError(
          "Numerics.hydro.ion_art_heat_C > 0 requires Numerics.hydro.av_type=\"vnr\"");
    }
  }
  if (numerics.hydro.hk_velocity_damper_C > 0.0) {
    if (main.dimension != "1D_SPH") {
      throw ConfigError(
          "Numerics.hydro.hk_velocity_damper_C > 0 is supported only in 1D_SPH");
    }
    if (numerics.hydro.hk_velocity_damper_C > 1.0) {
      throw ValueError("Numerics.hydro.hk_velocity_damper_C must be <= 1");
    }
  }
  ensure_int_ge(numerics.hydro.hk_velocity_damper_guard_cells, 0,
                "Numerics.hydro.hk_velocity_damper_guard_cells");
#if !TENRYU_ENABLE_HYPRE
  if (numerics.conduction.solver == "hypre") {
    throw ConfigError("Numerics.conduction.solver=\"hypre\" requires TENRYU_ENABLE_HYPRE=ON");
  }
#endif

  if (!(numerics.dt.cfl_hydro > 0.0 && numerics.dt.cfl_hydro <= 1.0)) {
    throw ValueError("Numerics.dt.cfl_hydro must be in (0, 1]");
  }
  if (!(numerics.dt.cfl_cond > 0.0 && numerics.dt.cfl_cond <= 1.0)) {
    throw ValueError("Numerics.dt.cfl_cond must be in (0, 1]");
  }
  if (numerics.ale.every_n_steps < 1) {
    throw ValueError("Numerics.ale.every_n_steps must be >= 1");
  }
  if (numerics.ale.force_rezone_every_n_steps < 0) {
    throw ValueError("Numerics.ale.force_rezone_every_n_steps must be >= 0");
  }
  if (numerics.ale.warmup_steps < 0) {
    throw ValueError("Numerics.ale.warmup_steps must be >= 0");
  }
  if (!(numerics.ale.relaxation >= 0.0 && numerics.ale.relaxation <= 1.0)) {
    throw ValueError("Numerics.ale.relaxation must be in [0, 1]");
  }
  if (!(numerics.ale.spacing_ratio_threshold >= 1.0)) {
    throw ValueError("Numerics.ale.spacing_ratio_threshold must be >= 1");
  }
  if (!(numerics.ale.quality_threshold > 0.0 && numerics.ale.quality_threshold <= 1.0)) {
    throw ValueError("Numerics.ale.quality_threshold must be in (0, 1]");
  }
  if (numerics.ale.max_iterations < 1) {
    throw ValueError("Numerics.ale.max_iterations must be >= 1");
  }
  if (!(numerics.ale.convergence_tol > 0.0)) {
    throw ValueError("Numerics.ale.convergence_tol must be > 0");
  }
  if (!(numerics.ale.max_displacement_fraction > 0.0 &&
        numerics.ale.max_displacement_fraction <= 1.0)) {
    throw ValueError("Numerics.ale.max_displacement_fraction must be in (0, 1]");
  }
  if (!is_ale_remap_limiter(numerics.ale.remap_limiter)) {
    throw ConfigError(
        "Numerics.ale.remap_limiter must be one of {\"van_leer\", \"minmod\"}");
  }
  if (numerics.ale.axis_z_motion != "fixed" &&
      numerics.ale.axis_z_motion != "winslow" &&
      numerics.ale.axis_z_motion != "lagrangian" &&
      numerics.ale.axis_z_motion != "lagrangian_tangential") {
    throw ValueError(
        "Numerics.ale.axis_z_motion must be one of {\"fixed\", \"winslow\", "
        "\"lagrangian\", \"lagrangian_tangential\"}");
  }
  if (numerics.ale.axis_z_motion == "lagrangian") {
    throw ValueError(
        "Numerics.ale.axis_z_motion=\"lagrangian\" is not yet implemented "
        "(Phase 8b future work) — use \"lagrangian_tangential\" instead");
  }
  if (!(numerics.ale.winslow_axis_kappa > 0.0 &&
        numerics.ale.winslow_axis_kappa <= 1.0)) {
    throw ValueError("Numerics.ale.winslow_axis_kappa must be in (0, 1]");
  }
  if (numerics.ale.reference_target != "none" &&
      numerics.ale.reference_target != "eulerian_initial" &&
      numerics.ale.reference_target != "spherical_equal_angle") {
    throw ValueError(
        "Numerics.ale.reference_target must be one of {\"none\", "
        "\"eulerian_initial\", \"spherical_equal_angle\"}");
  }
  if (!(numerics.ale.reference_blend_default >= 0.0 &&
        numerics.ale.reference_blend_default <= 1.0)) {
    throw ValueError("Numerics.ale.reference_blend_default must be in [0, 1]");
  }
  if (!(numerics.ale.reference_volume_floor_rel >= 0.0)) {
    throw ValueError("Numerics.ale.reference_volume_floor_rel must be >= 0");
  }
  if (!(numerics.ale.reference_corner_j_floor_rel >= 0.0)) {
    throw ValueError("Numerics.ale.reference_corner_j_floor_rel must be >= 0");
  }
  if (!(numerics.ale.reference_gauss_j_floor_rel >= 0.0)) {
    throw ValueError("Numerics.ale.reference_gauss_j_floor_rel must be >= 0");
  }
  if (numerics.ale.reference_linesearch_max_iters < 0 ||
      numerics.ale.reference_linesearch_max_iters > 60) {
    throw ValueError("Numerics.ale.reference_linesearch_max_iters must be in [0, 60]");
  }
  if (!(numerics.ale.reference_trigger_axis_margin_threshold >= 0.0)) {
    throw ValueError(
        "Numerics.ale.reference_trigger_axis_margin_threshold must be >= 0");
  }
  if (!(numerics.ale.reference_trigger_corner_j_ratio_threshold >= 0.0)) {
    throw ValueError(
        "Numerics.ale.reference_trigger_corner_j_ratio_threshold must be >= 0");
  }
  if (!(numerics.ale.remap_damage_dmax >= 0.0)) {
    throw ValueError("Numerics.ale.remap_damage_dmax must be >= 0");
  }
  if (!(numerics.ale.remap_damage_axis_eta >= 0.0)) {
    throw ValueError("Numerics.ale.remap_damage_axis_eta must be >= 0");
  }
  if (!(numerics.ale.remap_damage_axis_budget_factor >= 0.0)) {
    throw ValueError("Numerics.ale.remap_damage_axis_budget_factor must be >= 0");
  }
  if (numerics.ale.remap_damage_axis_budget_enabled &&
      !numerics.ale.remap_damage_gate_enabled) {
    throw ValueError(
        "Numerics.ale.remap_damage_axis_budget_enabled=true requires "
        "remap_damage_gate_enabled=true");
  }
  if (!(numerics.ale.predictive_acceptance_axis_floor_fraction >= 0.0 &&
        numerics.ale.predictive_acceptance_axis_floor_fraction <= 1.0)) {
    throw ValueError(
        "Numerics.ale.predictive_acceptance_axis_floor_fraction must be in [0, 1]");
  }
  if (!(numerics.ale.predictive_acceptance_cell_vol_floor_fraction >= 0.0 &&
        numerics.ale.predictive_acceptance_cell_vol_floor_fraction <= 1.0)) {
    throw ValueError(
        "Numerics.ale.predictive_acceptance_cell_vol_floor_fraction must be in [0, 1]");
  }
  if (numerics.ale.safe_backtrack_min_exp < 0 ||
      numerics.ale.safe_backtrack_min_exp > 60) {
    throw ValueError("Numerics.ale.safe_backtrack_min_exp must be in [0, 60]");
  }
  if (numerics.ale.safe_backtrack_binary_iters < 0 ||
      numerics.ale.safe_backtrack_binary_iters > 60) {
    throw ValueError("Numerics.ale.safe_backtrack_binary_iters must be in [0, 60]");
  }
  if (!(numerics.ale.corner_cell_aspect_eta >= 0.0 &&
        numerics.ale.corner_cell_aspect_eta <= 1.0)) {
    throw ValueError("Numerics.ale.corner_cell_aspect_eta must be in [0, 1]");
  }
  if (numerics.ale.rezone_solver != "legacy_winslow" &&
      numerics.ale.rezone_solver != "rz_full_metric_winslow" &&
      numerics.ale.rezone_solver != "m1_tmop") {
    throw ValueError(
        "Numerics.ale.rezone_solver must be one of "
        "{\"legacy_winslow\", \"rz_full_metric_winslow\", \"m1_tmop\"}");
  }
  if (!(numerics.ale.m1_gamma_align >= 0.0)) {
    throw ValueError("Numerics.ale.m1_gamma_align must be >= 0");
  }
  if (!(numerics.ale.m1_lambda_tether >= 0.0)) {
    throw ValueError("Numerics.ale.m1_lambda_tether must be >= 0");
  }
  if (!(numerics.ale.m1_theta_reg >= 0.0)) {
    throw ValueError("Numerics.ale.m1_theta_reg must be >= 0");
  }
  if (numerics.ale.m1_sweeps < 1) {
    throw ValueError("Numerics.ale.m1_sweeps must be >= 1");
  }
  if (!(numerics.ale.m1_barrier_beta >= 0.0)) {
    throw ValueError("Numerics.ale.m1_barrier_beta must be >= 0");
  }
  if (numerics.ale.rezone_solver == "m1_tmop" &&
      (mesh.logical_mesh_2d == "rectangular_rz" ||
       mesh.logical_mesh_2d == "cone_shell")) {
    throw ConfigError(
        "Numerics.ale.rezone_solver=\"m1_tmop\" is staged for polar-family "
        "logical meshes; rectangular/cone logical meshes are not supported "
        "in M1 v1");
  }
  if (!(numerics.ale.rezone_local_j_floor_rel >= 0.0)) {
    throw ValueError("Numerics.ale.rezone_local_j_floor_rel must be >= 0");
  }
  if (numerics.ale.rezone_local_linesearch_max_halves < 0 ||
      numerics.ale.rezone_local_linesearch_max_halves > 32) {
    throw ValueError(
        "Numerics.ale.rezone_local_linesearch_max_halves must be in [0, 32]");
  }
  if (!(numerics.ale.zero_gauss_j_floor_rel > 0.0)) {
    throw ValueError("Numerics.ale.zero_gauss_j_floor_rel must be > 0");
  }
  if (numerics.ale.lambda_sweep_target_cell_c < -1) {
    throw ValueError("Numerics.ale.lambda_sweep_target_cell_c must be >= -1");
  }
  if (numerics.ale.lambda_sweep_target_cell_i < -1) {
    throw ValueError("Numerics.ale.lambda_sweep_target_cell_i must be >= -1");
  }
  if (numerics.ale.lambda_sweep_target_cell_j < -1) {
    throw ValueError("Numerics.ale.lambda_sweep_target_cell_j must be >= -1");
  }
  if ((numerics.ale.lambda_sweep_target_cell_i >= 0) !=
      (numerics.ale.lambda_sweep_target_cell_j >= 0)) {
    throw ValueError(
        "Numerics.ale.lambda_sweep_target_cell_i and "
        "lambda_sweep_target_cell_j must both be set or both be -1");
  }
  if (numerics.ale.lambda_sweep_max_exp < 0 ||
      numerics.ale.lambda_sweep_max_exp > 1022) {
    throw ValueError("Numerics.ale.lambda_sweep_max_exp must be in [0, 1022]");
  }
  if (numerics.ale.axis_repair_mode != "full_winslow" &&
      numerics.ale.axis_repair_mode != "axis_spine_only" &&
      numerics.ale.axis_repair_mode != "axis_z_winslow" &&
      numerics.ale.axis_repair_mode != "none") {
    throw ValueError(
        "Numerics.ale.axis_repair_mode must be one of "
        "{\"full_winslow\", \"axis_spine_only\", \"axis_z_winslow\", \"none\"}");
  }
  if (numerics.ale.remap_scheme != "legacy_split" &&
      numerics.ale.remap_scheme != "ms2_moments") {
    throw ValueError(
        "Numerics.ale.remap_scheme must be one of "
        "{\"legacy_split\", \"ms2_moments\"}");
  }
  if (numerics.ale.remap_ms2_limiter != "van_leer" &&
      numerics.ale.remap_ms2_limiter != "barth_jespersen") {
    throw ValueError(
        "Numerics.ale.remap_ms2_limiter must be one of "
        "{\"van_leer\", \"barth_jespersen\"}");
  }
  if (numerics.ale.conservative_remap_target != "reference") {
    throw ValueError(
        "Numerics.ale.conservative_remap_target must be \"reference\"");
  }
  if (numerics.ale.conservative_remap_order != "first_order_donor" &&
      numerics.ale.conservative_remap_order != "second_order_van_leer") {
    throw ValueError(
        "Numerics.ale.conservative_remap_order must be one of "
        "{\"first_order_donor\", \"second_order_van_leer\"}");
  }
  if (numerics.ale.tri_fan_tracking_reference_mode != "legacy_lagging" &&
      numerics.ale.tri_fan_tracking_reference_mode != "seamless_converging") {
    throw ValueError(
        "Numerics.ale.tri_fan_tracking_reference_mode must be one of "
        "{\"legacy_lagging\", \"seamless_converging\"}");
  }
  if (!(numerics.ale.tri_fan_tracking_reference_omega >= 0.0 &&
        numerics.ale.tri_fan_tracking_reference_omega <= 1.0)) {
    throw ValueError(
        "Numerics.ale.tri_fan_tracking_reference_omega must be in [0, 1]");
  }
  if (!(numerics.ale.tri_fan_tracking_reference_beta > 0.0 &&
        numerics.ale.tri_fan_tracking_reference_beta <= 1.0)) {
    throw ValueError(
        "Numerics.ale.tri_fan_tracking_reference_beta must be in (0, 1]");
  }
  if (!(numerics.ale.tri_fan_tracking_reference_g0 > 0.0)) {
    throw ValueError(
        "Numerics.ale.tri_fan_tracking_reference_g0 must be > 0");
  }
  if (!(numerics.ale.tri_fan_tracking_reference_nu >= 0.0)) {
    throw ValueError(
        "Numerics.ale.tri_fan_tracking_reference_nu must be >= 0");
  }
  if (!(numerics.ale.tri_fan_tracking_reference_eps_v > 0.0)) {
    throw ValueError(
        "Numerics.ale.tri_fan_tracking_reference_eps_v must be > 0");
  }
  if (numerics.ale.conservative_remap_lagrangian_bulk_center_node_ring_max < 0) {
    throw ValueError(
        "Numerics.ale.conservative_remap_lagrangian_bulk_center_node_ring_max "
        "must be >= 0");
  }
  if (numerics.ale.central_pseudo_core_enabled) {
    if (main.dimension != "2D_RZ") {
      throw ConfigError(
          "Numerics.ale.central_pseudo_core_enabled=true is supported only in 2D_RZ");
    }
    const bool five_block =
        mesh.topology_scheme ==
            TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK ||
        mesh.topology_scheme ==
            TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK;
    const bool hybrid_trifan_cap =
        mesh.topology_scheme ==
            TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER &&
        mesh.polar_tier_center_kind == "trifan_cap";
    if (!five_block && !hybrid_trifan_cap) {
      throw ConfigError(
          "Numerics.ale.central_pseudo_core_enabled=true requires a 5-block "
          "multiblock center topology or "
          "Mesh.topology_scheme=\"multiblock_polar_tier_cart_center\" with "
          "Mesh.polar_tier_center_kind=\"trifan_cap\"");
    }
    if (numerics.materials.per_material_conservation_enabled) {
      throw ConfigError(
          "Numerics.ale.central_pseudo_core_enabled=true does not support "
          "per-material conservation in Exp1");
    }
    if (!(numerics.ale.central_pseudo_core_s_c > 0.0)) {
      throw ValueError(
          "Numerics.ale.central_pseudo_core_s_c must be > 0 when "
          "central_pseudo_core_enabled=true");
    }
  }
  if (!is_plic_normal_estimator(numerics.plic.normal_estimator)) {
    throw ConfigError(
        "Numerics.plic.normal_estimator must be one of "
        "{\"youngs\", \"LVIRA\", \"youngs_seeded_LVIRA\"}");
  }
  if (!is_plic_t0_volume_cut_method(numerics.plic.t0_volume_cut_method)) {
    throw ConfigError(
        "Numerics.plic.t0_volume_cut_method must be one of "
        "{\"centroid_only_legacy\", \"adaptive_subdivision_2x2\", "
        "\"adaptive_subdivision_3x3\"}");
  }
  if (!is_plic_per_cell_state(numerics.plic.material_interface_per_cell_state)) {
    throw ConfigError(
        "Numerics.plic.material_interface_per_cell_state must be one of "
        "{\"off\", \"sparse_on_degradation\", \"dense_debug\"}");
  }
  if (numerics.plic.t0_volume_cut_max_depth < 4 ||
      numerics.plic.t0_volume_cut_max_depth > 16) {
    throw ValueError("Numerics.plic.t0_volume_cut_max_depth must be in [4, 16]");
  }
  ensure_positive_finite(numerics.plic.t0_volume_cut_volfrac_tol,
                         "Numerics.plic.t0_volume_cut_volfrac_tol");
  ensure_positive_finite(numerics.plic.fast_path_threshold_min,
                         "Numerics.plic.fast_path_threshold_min");
  ensure_positive_finite(numerics.plic.fast_path_threshold_max,
                         "Numerics.plic.fast_path_threshold_max");
  ensure_positive_finite(numerics.plic.alpha_tolerance_rel,
                         "Numerics.plic.alpha_tolerance_rel");
  ensure_positive_finite(numerics.plic.thermodynamic_error_soft_threshold,
                         "Numerics.plic.thermodynamic_error_soft_threshold");
  ensure_positive_finite(numerics.plic.thermodynamic_error_hard_threshold,
                         "Numerics.plic.thermodynamic_error_hard_threshold");
  ensure_positive_finite(numerics.plic.class_d_dense_fraction_threshold,
                         "Numerics.plic.class_d_dense_fraction_threshold");
  ensure_positive_finite(numerics.plic.drift_sensor_max_relative,
                         "Numerics.plic.drift_sensor_max_relative");
  ensure_positive_finite(numerics.plic.drift_sensor_max_swept_fraction,
                         "Numerics.plic.drift_sensor_max_swept_fraction");
  ensure_positive_finite(numerics.plic.prev_normal_freshness_volfrac_threshold,
                         "Numerics.plic.prev_normal_freshness_volfrac_threshold");
  ensure_positive_finite(numerics.plic.plic_per_step_cost_target_fraction,
                         "Numerics.plic.plic_per_step_cost_target_fraction");
  if (numerics.plic.fast_path_halo_radius_cells < 1) {
    throw ValueError("Numerics.plic.fast_path_halo_radius_cells must be >= 1");
  }
  if (numerics.plic.alpha_solver_max_iter < 1) {
    throw ValueError("Numerics.plic.alpha_solver_max_iter must be >= 1");
  }
  if (numerics.plic.enabled) {
    if (!(numerics.plic.fast_path_threshold_max < 1.0)) {
      throw ValueError("Numerics.plic.fast_path_threshold_max must be < 1 when PLIC is enabled");
    }
    if (numerics.plic.material_interface_per_cell_state == "dense_debug") {
      tenryu::core::log_warning(
          "Numerics.plic.material_interface_per_cell_state=\"dense_debug\" may substantially increase HDF5 output size");
    }
  }
  ensure_positive_finite(numerics.materials.presence_threshold_volfrac,
                         "Numerics.materials.presence_threshold_volfrac");
  ensure_positive_finite(
      numerics.materials.presence_threshold_mass_density_g_per_cc,
      "Numerics.materials.presence_threshold_mass_density_g_per_cc");
  ensure_positive_finite(
      numerics.materials.conservation_residual_warn_threshold_rel,
      "Numerics.materials.conservation_residual_warn_threshold_rel");
  ensure_positive_finite(
      numerics.materials.conservation_residual_hard_warning_threshold_rel,
      "Numerics.materials.conservation_residual_hard_warning_threshold_rel");
  if (numerics.materials.conservation_residual_hard_warning_threshold_rel <
      numerics.materials.conservation_residual_warn_threshold_rel) {
    throw ValueError(
        "Numerics.materials.conservation_residual_hard_warning_threshold_rel "
        "must be >= conservation_residual_warn_threshold_rel");
  }
  if (numerics.materials.deposit_redistribute_provenance_label.empty()) {
    throw ValueError(
        "Numerics.materials.deposit_redistribute_provenance_label must be non-empty");
  }
  for (const auto& [name, lower] :
       numerics.materials.eos_table_validity_lower_bound_g_per_cc) {
    ensure_positive_finite(
        lower,
        "Numerics.materials.eos_table_validity_lower_bound_g_per_cc." + name);
  }
  if (numerics.ale.axis_band_managed_remap_width < 1) {
    throw ValueError("Numerics.ale.axis_band_managed_remap_width must be >= 1");
  }
  if (numerics.ale.axis_band_managed_remap_max_width <
      numerics.ale.axis_band_managed_remap_width) {
    throw ValueError(
        "Numerics.ale.axis_band_managed_remap_max_width must be >= "
        "axis_band_managed_remap_width");
  }
  if (numerics.ale.axis_band_managed_remap_max_width > 32) {
    throw ValueError("Numerics.ale.axis_band_managed_remap_max_width must be <= 32");
  }
  if (!(numerics.ale.axis_band_managed_remap_margin_trigger > 0.0)) {
    throw ValueError(
        "Numerics.ale.axis_band_managed_remap_margin_trigger must be > 0");
  }
  if (numerics.ale.multiblock_differential_reference_band_count < 8 ||
      numerics.ale.multiblock_differential_reference_band_count > 4096) {
    throw ValueError(
        "Numerics.ale.multiblock_differential_reference_band_count must be in [8, 4096]");
  }
  if (!(numerics.ale.multiblock_differential_reference_smoothing_g0 > 0.0 &&
        numerics.ale.multiblock_differential_reference_smoothing_g0 <= 1.0)) {
    throw ValueError(
        "Numerics.ale.multiblock_differential_reference_smoothing_g0 must be in (0, 1]");
  }
  if (!(numerics.ale.multiblock_differential_reference_nu > 0.0 &&
        numerics.ale.multiblock_differential_reference_nu <= 1.0)) {
    throw ValueError(
        "Numerics.ale.multiblock_differential_reference_nu must be in (0, 1]");
  }
  if (!(numerics.ale.multiblock_differential_reference_eps_v > 0.0 &&
        numerics.ale.multiblock_differential_reference_eps_v <= 1.0)) {
    throw ValueError(
        "Numerics.ale.multiblock_differential_reference_eps_v must be in (0, 1]");
  }
  if (!(numerics.ale.multiblock_differential_reference_s_cap_min_rel > 0.0 &&
        numerics.ale.multiblock_differential_reference_s_cap_min_rel <= 1.0)) {
    throw ValueError(
        "Numerics.ale.multiblock_differential_reference_s_cap_min_rel must be in (0, 1]");
  }
  if (!(numerics.ale.multiblock_differential_reference_xi_seam_tol > 0.0 &&
        numerics.ale.multiblock_differential_reference_xi_seam_tol <= 1.0e-3)) {
    throw ValueError(
        "Numerics.ale.multiblock_differential_reference_xi_seam_tol must be in (0, 1e-3]");
  }
  if (!(numerics.ale.multiblock_differential_reference_sigma_warn_floor > 0.0 &&
        numerics.ale.multiblock_differential_reference_sigma_warn_floor <= 1.0)) {
    throw ValueError(
        "Numerics.ale.multiblock_differential_reference_sigma_warn_floor must be in (0, 1]");
  }
  if (!(numerics.ale.axis_rezone_trigger_edge_fraction > 0.0 &&
        numerics.ale.axis_rezone_trigger_edge_fraction <= 1.0)) {
    throw ValueError(
        "Numerics.ale.axis_rezone_trigger_edge_fraction must be in (0, 1]");
  }
  if (!(numerics.ale.axis_rezone_trigger_min_altitude_fraction > 0.0 &&
        numerics.ale.axis_rezone_trigger_min_altitude_fraction <= 1.0)) {
    throw ValueError(
        "Numerics.ale.axis_rezone_trigger_min_altitude_fraction must be in (0, 1]");
  }
  if (!(numerics.ale.axis_rezone_eta_floor > 0.0 &&
        numerics.ale.axis_rezone_eta_floor < 1.0)) {
    throw ValueError("Numerics.ale.axis_rezone_eta_floor must be in (0, 1)");
  }
  if (numerics.ale.remap_ms_post_max_iter < 1) {
    throw ValueError("Numerics.ale.remap_ms_post_max_iter must be >= 1");
  }
  if (!(numerics.ale.remap_ms_rescale_floor >= 0.0 &&
        numerics.ale.remap_ms_rescale_floor <= 1.0)) {
    throw ValueError("Numerics.ale.remap_ms_rescale_floor must be in [0, 1]");
  }
  if (numerics.ale.shock_sensor_guard_cells < 0) {
    throw ValueError("Numerics.ale.shock_sensor_guard_cells must be >= 0");
  }
  if (!(numerics.ale.density_jump_threshold >= 0.0)) {
    throw ValueError("Numerics.ale.density_jump_threshold must be >= 0");
  }
  if (!(numerics.ale.Te_jump_threshold >= 0.0)) {
    throw ValueError("Numerics.ale.Te_jump_threshold must be >= 0");
  }
  if (!(numerics.ale.preventive_axis_guard_fraction >= 0.0)) {
    throw ValueError("Numerics.ale.preventive_axis_guard_fraction must be >= 0");
  }
  if (numerics.ale.euler_window.axis_core_transaction_mode ==
          "clearance_replay" &&
      numerics.ale.euler_window.replay_table_path.empty()) {
    throw ConfigError(
        "Numerics.ale.euler_window "
        "axis_core_transaction_mode=\"clearance_replay\" requires "
        "a non-empty replay_table_path");
  }
  validate_euler_window_config(
      numerics.ale.euler_window, "Numerics.ale.euler_window");
  for (std::size_t i = 0; i < numerics.ale.euler_windows.size(); ++i) {
    validate_euler_window_config(
        numerics.ale.euler_windows[i],
        "Numerics.ale.euler_windows[" + std::to_string(i) + "]");
  }
  validate_band_ale_config(
      numerics.ale.band_ale, "Numerics.ale.band_ale");
  tenryu::core::validate_band_ale_config(config);
  tenryu::core::validate_closure_catchment_config(config);
  tenryu::core::validate_pole_theta_config(config);
  validate_evacuated_cell_config(
      numerics.ale.evacuated_cell, "Numerics.ale.evacuated_cell");
  const bool axis_survival_core_configured =
      numerics.ale.euler_window.role == "axis_survival_core";
  const bool axis_survival_core_enabled =
      numerics.ale.euler_window.enabled && axis_survival_core_configured;
  if (numerics.ale.euler_window.axis_core_transaction_mode != "static" &&
      !main.restart_from.empty()) {
    throw ConfigError(
        "Numerics.ale.euler_window "
        "axis_core_transaction_mode=\"" +
        numerics.ale.euler_window.axis_core_transaction_mode +
        "\" rejects "
        "Main.restart_from in v1");
  }
  if (axis_survival_core_configured &&
      !numerics.ale.euler_windows.empty()) {
    throw ConfigError(
        "Numerics.ale.euler_window role=\"axis_survival_core\" requires "
        "Numerics.ale.euler_windows to be empty");
  }
  for (std::size_t i = 0; i < numerics.ale.euler_windows.size(); ++i) {
    if (numerics.ale.euler_windows[i].role == "axis_survival_core") {
      throw ConfigError(
          "axis_survival_core must use the single "
          "Numerics.ale.euler_window entry");
    }
  }
  if (axis_survival_core_configured &&
      !numerics.ale.conservative_remap_enabled) {
    throw ConfigError(
        "Numerics.ale.euler_window role=\"axis_survival_core\" requires "
        "Numerics.ale.conservative_remap_enabled=true");
  }
  if (mesh.polar_tier_dendrite_enabled &&
      numerics.ale.rezone_solver == "m1_tmop") {
    throw ConfigError(
        "Mesh.polar_tier_dendrite_enabled=true is mutually exclusive with "
        "Numerics.ale.rezone_solver=\"m1_tmop\" in v1: the M1 theta-ring "
        "machinery is not qualified on nonuniform dendrite ladders "
        "(tmp/ale_p2_briefs/design_dendrite.md §7)");
  }
  if (mesh.polar_tier_dendrite_enabled &&
      (numerics.ale.euler_window.enabled ||
       !numerics.ale.euler_windows.empty()) &&
      !axis_survival_core_enabled) {
    throw ConfigError(
        "Mesh.polar_tier_dendrite_enabled=true is mutually exclusive with "
        "Numerics.ale.euler_window.enabled=true or non-empty "
        "Numerics.ale.euler_windows unless the single Euler window has "
        "role=\"axis_survival_core\" in v1: the plain Euler-window machinery is "
        "not qualified on nonuniform dendrite ladders "
        "(tmp/ale_p2_briefs/design_dendrite.md §7)");
  }
  if (numerics.ale.euler_window.enabled ||
      !numerics.ale.euler_windows.empty()) {
    if (!numerics.ale.conservative_remap_enabled) {
      throw ConfigError(
          "Numerics.ale.euler_window.enabled=true requires "
          "Numerics.ale.conservative_remap_enabled=true");
    }
    if (numerics.ale.rezone_solver == "m1_tmop") {
      throw ConfigError(
          "Numerics.ale.euler_window.enabled=true is mutually exclusive "
          "with Numerics.ale.rezone_solver=\"m1_tmop\"");
    }
    if (!tenryu::core::is_multiblock_differential_reference_topology(config)) {
      throw ConfigError(
          "Numerics.ale.euler_window.enabled=true requires "
          "Main.dimension=\"2D_RZ\" and multiblock Mesh.topology_scheme");
    }
  }
  if (numerics.ale.band_ale.enabled) {
    if (numerics.ale.band_ale.compose_with_rezone &&
        numerics.ale.band_ale.bands != "estimator") {
      throw ConfigError(
          "Numerics.ale.band_ale.compose_with_rezone=true requires "
          "Numerics.ale.band_ale.bands=\"estimator\"");
    }
    if (numerics.ale.band_ale.compose_with_rezone &&
        config.mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK) {
      throw ConfigError(
          "band_ale.compose_with_rezone is single-block only (§20 v1)");
    }
    if (numerics.ale.band_ale.bands == "estimator" &&
        !numerics.diagnostics.refinement_estimator.enabled) {
      throw ConfigError(
          "Numerics.ale.band_ale.bands=\"estimator\" requires "
          "Numerics.diagnostics.refinement_estimator.enabled=true");
    }
    if (numerics.ale.band_ale.bands == "estimator" &&
        !numerics.diagnostics.refinement_autopilot.enabled) {
      throw ConfigError(
          "Numerics.ale.band_ale.bands=\"estimator\" requires "
          "Diagnostics.refinement_autopilot.enabled (the §18.5 front hold "
          "needs the tracker)");
    }
    if (!numerics.ale.conservative_remap_enabled) {
      throw ConfigError(
          "Numerics.ale.band_ale.enabled=true requires "
          "Numerics.ale.conservative_remap_enabled=true");
    }
    if (numerics.ale.rezone_solver == "m1_tmop") {
      throw ConfigError(
          "Numerics.ale.band_ale.enabled=true is mutually exclusive "
          "with Numerics.ale.rezone_solver=\"m1_tmop\"");
    }
    if ((numerics.ale.euler_window.enabled ||
         !numerics.ale.euler_windows.empty()) &&
        !axis_survival_core_enabled) {
      throw ConfigError(
          "Numerics.ale.band_ale.enabled=true is mutually exclusive "
          "with Numerics.ale.euler_window");
    }
    const bool band_ale_topology_ok =
        config.main.dimension == "2D_RZ" &&
        (config.mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK ||
         numerics.ale.band_ale.bands == "estimator");
    if (!band_ale_topology_ok) {
      throw ConfigError(
          "Numerics.ale.band_ale.enabled=true requires Main.dimension=\"2D_RZ\" "
          "and a multiblock Mesh.topology_scheme (bands=\"estimator\" also "
          "accepts single_block, §18.6)");
    }
  }
  if (numerics.ale.evacuated_cell.enabled) {
    if (main.dim != 2) {
      throw ConfigError(
          "Numerics.ale.evacuated_cell requires Main.dimension=\"2D_RZ\"");
    }
    if (mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK) {
      throw ConfigError(
          "Numerics.ale.evacuated_cell is single-block only (§21 v2a)");
    }
    if (!numerics.ale.enabled) {
      throw ConfigError(
          "Numerics.ale.evacuated_cell requires Numerics.ale.enabled=true "
          "(the policy is for ALE-managed exteriors)");
    }
    if (!numerics.diagnostics.conduction_energy_rate_export.enabled) {
      throw ConfigError(
          "Numerics.ale.evacuated_cell requires conduction_energy_rate_export "
          "(the coupling gate consumes it)");
    }
    if (numerics.conduction.nonlocal_model == "snb") {
      throw ConfigError(
          "evacuated_cell with SNB nonlocal conduction is not supported yet "
          "(§21 v2a is local-conduction only)");
    }
  }
  tenryu::core::validate_ale1d_config(config);
  tenryu::core::validate_dt_config(config);
  tenryu::core::validate_reale_v2_config(config);
  tenryu::core::validate_z_reflection_config(config);
  tenryu::core::validate_ale_identity_diag_config(config);
  tenryu::core::validate_transaction_failure_inject_config(config);
  tenryu::core::validate_m1_tmop_config(config);
  tenryu::core::validate_multiblock_differential_reference_config(config);
  tenryu::core::validate_central_pseudo_core_config(config);
  tenryu::core::validate_polar_tier_runtime_config(config);
  tenryu::core::validate_icf_standard_ale_profile_config(config);
  tenryu::core::validate_legacy_regression_profile(config);
  validate_production_audit_config(numerics);
  ensure_non_negative(numerics.diagnostics.icf.rho_inner_threshold_g_per_cc,
                      "Numerics.diagnostics.icf.rho_inner_threshold_g_per_cc");
  ensure_non_negative(numerics.diagnostics.icf.rho_outer_threshold_g_per_cc,
                      "Numerics.diagnostics.icf.rho_outer_threshold_g_per_cc");
  if (numerics.diagnostics.hotspot_gas.enabled) {
    if (!(numerics.diagnostics.hotspot_gas.R_g_cm > 0.0)) {
      throw ValueError("Numerics.diagnostics.hotspot_gas.R_g_cm must be > 0 when enabled");
    }
    if (!(numerics.diagnostics.hotspot_gas.mass_drift_warn_rel >= 0.0)) {
      throw ValueError(
          "Numerics.diagnostics.hotspot_gas.mass_drift_warn_rel must be >= 0");
    }
  }
  if (numerics.diagnostics.ale_velcoherence.every_n_steps < 1) {
    throw ValueError(
        "Numerics.diagnostics.ale_velcoherence.every_n_steps must be >= 1");
  }
  if (numerics.diagnostics.mesh_degeneracy_forensics.same_cell_count < 1) {
    throw ValueError(
        "Numerics.diagnostics.mesh_degeneracy_forensics.same_cell_count must be >= 1");
  }
  if (!(numerics.diagnostics.mesh_degeneracy_forensics.sigma_threshold > 0.0 &&
        numerics.diagnostics.mesh_degeneracy_forensics.sigma_threshold <= 1.0)) {
    throw ValueError(
        "Numerics.diagnostics.mesh_degeneracy_forensics.sigma_threshold must be in (0, 1]");
  }
  if (numerics.diagnostics.mesh_degeneracy_forensics.max_dumps_per_run < 0) {
    throw ValueError(
        "Numerics.diagnostics.mesh_degeneracy_forensics.max_dumps_per_run must be >= 0");
  }
  if (numerics.diagnostics.mesh_degeneracy_forensics
          .velocity_history_target_cell_c < -1) {
    throw ValueError(
        "Numerics.diagnostics.mesh_degeneracy_forensics."
        "velocity_history_target_cell_c must be >= -1");
  }
  if (numerics.diagnostics.mesh_degeneracy_forensics
          .velocity_history_sample_every_n_steps < 1) {
    throw ValueError(
        "Numerics.diagnostics.mesh_degeneracy_forensics."
        "velocity_history_sample_every_n_steps must be >= 1");
  }
  if (numerics.diagnostics.mesh_degeneracy_forensics
          .velocity_history_max_records < 0) {
    throw ValueError(
        "Numerics.diagnostics.mesh_degeneracy_forensics."
        "velocity_history_max_records must be >= 0");
  }
  if (!(numerics.dt.f_min_fleck > 0.0 && numerics.dt.f_min_fleck <= 1.0)) {
    throw ValueError("Numerics.dt.f_min_fleck must be in (0, 1]");
  }
  if (radiation.imc.f_max < numerics.dt.f_min_fleck) {
    throw ConfigError("f_max < f_min_fleck is invalid");
  }

  ensure_non_negative(mesh.floors.rho_floor_gcc, "Mesh.floors.rho_floor_gcc");
  ensure_non_negative(mesh.floors.Te_floor_eV, "Mesh.floors.Te_floor_eV");
  ensure_non_negative(mesh.floors.Ti_floor_eV, "Mesh.floors.Ti_floor_eV");
  ensure_non_negative(numerics.floors.rho, "Numerics.floors.rho");
  ensure_non_negative(numerics.floors.Te, "Numerics.floors.Te");
  ensure_non_negative(numerics.floors.Ti, "Numerics.floors.Ti");
  if (config.output.directory.empty()) {
    throw ConfigError("Output.directory must not be empty");
  }
  if (!is_output_format(config.output.format)) {
    throw ConfigError("Output.format must be \"hdf5\"");
  }
  if (!is_output_compression(config.output.compression)) {
    throw ConfigError("Output.compression must be \"none\" or \"gzip\"");
  }
  if (config.output.plot_every < 0) {
    throw ValueError("Output.plot_every must be >= 0");
  }
  if (config.output.history_every < 0) {
    throw ValueError("Output.history_every must be >= 0");
  }
  if (config.output.checkpoint_every < 0) {
    throw ValueError("Output.checkpoint_every must be >= 0");
  }
  if (config.output.checkpoint_keep_last < 0) {
    throw ConfigError("Output.checkpoint_keep_last must be >= 0");
  }
  if (config.output.compression_level < 0 || config.output.compression_level > 9) {
    throw ConfigError("Output.compression_level must be in [0, 9]");
  }

  if (parallel.decomposition.min_cells_per_rank < 1) {
    throw ConfigError("Parallel.decomposition.min_cells_per_rank must be >= 1");
  }
  for (std::size_t i = 0; i < parallel.decomposition.dims.size(); ++i) {
    if (parallel.decomposition.dims[i] < 1) {
      throw ConfigError("Parallel.decomposition.dims[" + std::to_string(i) +
                        "] must be >= 1");
    }
  }
  if (parallel.halo.ghost_layers < 1) {
    throw ConfigError("Parallel.halo.ghost_layers must be >= 1");
  }
  if (parallel.migration.max_substeps < 1) {
    throw ConfigError("Parallel.migration.max_substeps must be >= 1");
  }
  if (parallel.migration.emigrant_threshold < 1) {
    throw ConfigError("Parallel.migration.emigrant_threshold must be >= 1");
  }
  if (parallel.migration.initial_capacity < 1) {
    throw ConfigError("Parallel.migration.initial_capacity must be >= 1");
  }

  // dt cross-field consistency
  if (!(numerics.dt.initial_s > 0.0 || numerics.dt.initial_s < 0.0)) {
    throw ValueError("Numerics.dt.initial_s must be > 0 or None");
  }
  if (!(numerics.dt.min_s > 0.0)) {
    throw ValueError("Numerics.dt.min_s must be > 0");
  }
  if (!(numerics.dt.max_s > 0.0)) {
    throw ValueError("Numerics.dt.max_s must be > 0");
  }
  if (numerics.dt.min_s > numerics.dt.max_s) {
    throw ConfigError("Numerics.dt.min_s must be <= max_s");
  }
  if (!(numerics.dt.growth_factor >= 1.0)) {
    throw ValueError("Numerics.dt.growth_factor must be >= 1.0");
  }
  if (numerics.dt.growth_factor <= 1.0) {
    tenryu::core::log_warning(
        "Numerics.dt.growth_factor <= 1.0 makes every transient dt reduction "
        "permanent (dt can never recover); encode a fixed-dt intent via "
        "dt.max_s instead");
  }
  if (numerics.dt.floor_stall_max_consecutive_steps < 0) {
    throw ValueError("Numerics.dt.floor_stall_max_consecutive_steps must be >= 0");
  }

  // opacity clamp consistency
  if (numerics.safety.opacity_floor > numerics.safety.opacity_cap) {
    throw ConfigError("Numerics.safety.opacity_floor must be <= opacity_cap");
  }

  const auto validate_time_callable_numeric =
      [&](const std::string& callable_path, const py::object& callable_obj) {
        std::array<double, 3> sample_times = {
            0.0,
            0.5 * main.t_end,
            main.t_end};
        for (const double t_sample : sample_times) {
          py::object result;
          try {
            result = callable_obj(t_sample);
          } catch (const py::error_already_set& e) {
            std::ostringstream oss;
            oss << callable_path << " callable raised exception at t=" << t_sample
                << ": " << e.what();
            throw ConfigError(oss.str());
          } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << callable_path << " callable raised exception at t=" << t_sample
                << ": " << e.what();
            throw ConfigError(oss.str());
          }

          if (result.is_none() || py::isinstance<py::bool_>(result)) {
            std::ostringstream oss;
            oss << callable_path
                << " callable must return int/float, got " << py_type_name(result)
                << " at t=" << t_sample;
            throw ConfigError(oss.str());
          }

          double numeric_result = 0.0;
          try {
            numeric_result = py::cast<double>(result);
          } catch (const py::cast_error&) {
            std::ostringstream oss;
            oss << callable_path
                << " callable must return int/float, got " << py_type_name(result)
                << " at t=" << t_sample;
            throw ConfigError(oss.str());
          }

          if (!std::isfinite(numeric_result)) {
            std::ostringstream oss;
            oss << callable_path << " callable returned non-finite value at t="
                << t_sample << ": " << numeric_result;
            throw ConfigError(oss.str());
          }
        }
      };

  // laser beam completeness (direction or theta/phi required; power required)
  if (laser.enabled) {
    for (std::size_t i = 0; i < laser.beams.size(); ++i) {
      const auto& beam = laser.beams[i];
      const std::string profile_model =
          beam.profile_model.empty() ? laser.profile_model : beam.profile_model;
      if (profile_model == "custom") {
        throw ConfigError("Laser.beams[" + std::to_string(i) +
                          "] profile.model=\"custom\" is not implemented in M12");
      }
      if (profile_model != "gaussian" && profile_model != "super_gaussian" &&
          profile_model != "flat_top" && profile_model != "table") {
        throw ConfigError("Laser.beams[" + std::to_string(i) +
                          "] has unknown profile.model=\"" + profile_model +
                          "\" (allowed: gaussian, super_gaussian, flat_top, table)");
      }
      const bool has_direction = !beam.direction.empty();
      const bool has_theta = !std::isnan(beam.theta);
      const bool has_phi = !std::isnan(beam.phi);
      if (!has_direction && !(has_theta && has_phi)) {
        throw ConfigError("Laser.beams[" + std::to_string(i) +
                          "] requires direction vector or both theta and phi");
      }
      if (!beam.focus.empty() && beam.focus.size() != 3) {
        throw ConfigError("Laser.beams[" + std::to_string(i) +
                          "].focus must contain exactly 3 values");
      }
      if (!beam.power.detected) {
        throw ConfigError("Laser.beams[" + std::to_string(i) +
                          "].power must be a callable");
      }
    }
  }
  for (std::size_t i = 0; i < laser.beams.size(); ++i) {
    if (!laser.beams[i].power.detected) {
      continue;
    }
    const std::string path = "Laser.beams[" + std::to_string(i) + "].power";
    const auto callable_it = callable_objects.find(path);
    if (callable_it == callable_objects.end()) {
      throw ConfigError("Missing callable object for " + path);
    }
    validate_time_callable_numeric(path, callable_it->second);
  }

  if (radiation.boundary.marshak_Tr.detected) {
    const std::string path = "Radiation.boundary.marshak_Tr";
    const auto callable_it = callable_objects.find(path);
    if (callable_it == callable_objects.end()) {
      throw ConfigError("Missing callable object for " + path);
    }
    validate_time_callable_numeric(path, callable_it->second);
  }
  for (const auto& [face, _] : radiation.boundary.marshak_Tr_map) {
    const std::string path = "Radiation.boundary.marshak_Tr_map." + face;
    const auto callable_it = callable_objects.find(path);
    if (callable_it == callable_objects.end()) {
      throw ConfigError("Missing callable object for " + path);
    }
    validate_time_callable_numeric(path, callable_it->second);
  }
  if (numerics.hydro.pressure_drive_1d.detected) {
    const std::string path = "Numerics.hydro.boundary_pressure";
    const auto callable_it = callable_objects.find(path);
    if (callable_it == callable_objects.end()) {
      throw ConfigError("Missing callable object for " + path);
    }
    validate_time_callable_numeric(path, callable_it->second);
  }

  if (radiation.boundary.marshak_Tr_eV >= 0.0 &&
      !(radiation.boundary.marshak_Tr_eV > 0.0)) {
    throw ValueError("Radiation.boundary.marshak_Tr_eV must be > 0");
  }

  // marshak boundary requires callable or constant temperature.
  const auto check_marshak_callable = [&](const std::string& face_type,
                                           std::string_view face_name) {
    if (face_type == "marshak") {
      if (!radiation.boundary.marshak_Tr.detected &&
          !radiation.boundary.marshak_Tr_map.contains(std::string(face_name)) &&
          !(radiation.boundary.marshak_Tr_eV > 0.0)) {
        throw ConfigError("Radiation boundary face '" + std::string(face_name) +
                          "' is marshak but no marshak_Tr/marshak_Tr_map/marshak_Tr_eV provided");
      }
    }
  };
  if (main.dim == 2 && radiation.boundary.outer_r == "marshak") {
    throw ConfigError("Radiation.boundary.r_outer='marshak' is not supported in 2D_RZ; only z_bottom/z_top support marshak");
  }

  if (radiation.enabled) {
    if (main.dim == 1 && numerics.hydro.enabled && mesh.r_min == 0.0 &&
        radiation.boundary.inner_r != "reflect") {
      throw ConfigError("1D_SPH requires Radiation.boundary.inner_r=\"reflect\"");
    }
    check_marshak_callable(radiation.boundary.type, "type");
    check_marshak_callable(radiation.boundary.inner_r, "inner_r");
    check_marshak_callable(radiation.boundary.outer_r, "outer_r");
    if (main.dim == 2) {
      check_marshak_callable(radiation.boundary.bottom_z, "bottom_z");
      check_marshak_callable(radiation.boundary.top_z, "top_z");
    }
  }

  // pressure boundary requires callable
  if (main.dim == 1 && numerics.hydro.boundary_1d == "axis") {
    throw ConfigError("Numerics.hydro.boundary_1d=\"axis\" is not supported in 1D_SPH");
  }
  if (main.dim == 1 && numerics.hydro.boundary_1d == "state_supply") {
    throw ConfigError("Numerics.hydro.boundary_1d=\"state_supply\" is not supported");
  }
  if (numerics.hydro.boundary_1d == "pressure" &&
      !numerics.hydro.pressure_drive_1d.detected) {
    throw ConfigError(
        "Numerics.hydro.boundary='pressure' requires boundary_pressure callable");
  }
  if (main.dim == 2) {
    auto& b2d = numerics.hydro.boundary_2d;
    if (!is_mesh_tangential_target(b2d.mesh_tangential_target)) {
      throw ConfigError(
          "Numerics.hydro.boundary_2d.mesh_tangential_target must be "
          "\"lagrangian\" or \"reference\"");
    }
    if (!is_state_supply_donor_mode(b2d.state_supply_donor_mode)) {
      throw ConfigError(
          "Numerics.hydro.boundary_2d.state_supply_donor_mode must be "
          "\"interior_per_i\" or \"interior_radial_average\"");
    }
    const bool r_min_is_axis_for_boundary =
        std::abs(mesh.r_min) <= numerics.axis_eps_cm &&
        mesh.topology_scheme != core::TopologyScheme::PENTAGON_BELT_SHELL;
    if (r_min_is_axis_for_boundary) {
      // For physical-axis case (r_min ~= 0), r_inner must be "axis".
      if (b2d.r_inner != "axis") {
        throw ConfigError(
            "Numerics.hydro.boundary_2d.r_inner must be \"axis\" in 2D_RZ "
            "when Mesh.r_min == 0");
      }
    } else {
      // For annular case (r_min > 0), r_inner must be a reflective or pinned
      // wall. The "axis" literal is still accepted (legacy) but warns that
      // its semantics are reflective.
      if (b2d.r_inner != "axis" && b2d.r_inner != "reflect" &&
          b2d.r_inner != "pinned") {
        throw ConfigError(
            "Numerics.hydro.boundary_2d.r_inner must be \"reflect\" (preferred) "
            "or \"axis\" (legacy) or \"pinned\" in 2D_RZ when Mesh.r_min > 0");
      }
      if (b2d.r_inner == "axis") {
        tenryu::core::log_warning(
            "Numerics.hydro.boundary_2d.r_inner=\"axis\" with Mesh.r_min > 0: "
            "semantically a reflective inner wall (annular); "
            "prefer \"reflect\" for clarity");
      }
      if (numerics.ale.axis_repair_mode != "none") {
        tenryu::core::log_warning(
            "Numerics.ale.axis_repair_mode=\"" + numerics.ale.axis_repair_mode +
            "\" with Mesh.r_min > 0 is meaningless for annular 2D_RZ and is "
            "internally treated as \"none\"");
      }
    }
    if (b2d.r_inner == "state_supply" || b2d.r_outer == "state_supply") {
      throw ConfigError(
          "Numerics.hydro.boundary_2d state_supply is supported only on z_bottom/z_top");
    }
    if (b2d.z_bottom == "pressure" || b2d.z_top == "pressure") {
      throw ConfigError(
          "Numerics.hydro.boundary_2d.z_bottom/z_top='pressure' is not supported");
    }
    // State-supply mesh nodes are ALE-safe via mesh-velocity/material-velocity
    // decoupling.  ALE rezone of interior nodes is allowed; the
    // boundary node stays at z_min/z_max with v_z_mesh = 0.
    if (b2d.r_outer == "pressure" && !numerics.hydro.pressure_drive_1d.detected) {
      throw ConfigError(
          "Numerics.hydro.boundary_2d.r_outer='pressure' requires boundary_pressure callable");
    }
    b2d.sync_legacy_strings();
  }
  const bool is_2d_rz = (main.dimension == "2D_RZ");
  const bool r_min_is_axis = std::abs(mesh.r_min) <= numerics.axis_eps_cm;
  const bool r_inner_is_axis_like =
      (numerics.hydro.boundary_2d.r_inner == "axis");
  numerics.has_physical_rz_axis =
      is_2d_rz && r_min_is_axis && r_inner_is_axis_like;

  if (config.output.plot_every_s > 0.0 && !config.output.plot_every_explicit) {
    config.output.plot_every = 0;
  }
  if (config.output.history_every_s > 0.0 &&
      !config.output.history_every_explicit) {
    config.output.history_every = 0;
  }
  if (config.output.checkpoint_every_s > 0.0 &&
      !config.output.checkpoint_every_explicit) {
    config.output.checkpoint_every = 0;
  }
}

void begin_build(Builder& builder) {
  if (g_active_builder != nullptr) {
    throw ConfigError("begin_build called while another Builder is active");
  }
  g_active_builder = &builder;
}

void end_build() {
  g_active_builder = nullptr;
}

Builder* active_builder() {
  return g_active_builder;
}

Builder& require_active_builder() {
  if (g_active_builder == nullptr) {
    throw ConfigError("tenryu_namelist block called outside Runtime::execute()");
  }
  return *g_active_builder;
}

}  // namespace tenryu::core::namelist

#else

namespace tenryu::core::namelist {

void begin_build(Builder&) {}
void end_build() {}
Builder* active_builder() {
  return nullptr;
}
Builder& require_active_builder() {
  throw ConfigError("Python support is disabled");
}

}  // namespace tenryu::core::namelist

#endif
