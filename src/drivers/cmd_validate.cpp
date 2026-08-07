#include "drivers/cli.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#if TENRYU_ENABLE_PYTHON
#include <pybind11/pybind11.h>
#include "core/namelist/runtime.hpp"
#endif

namespace tenryu::drivers {
namespace {

std::string format_cli_error(const std::string& message) {
  return "TENRYU ERROR [namelist]: " + message;
}

template <typename T>
std::string value_string(const T& value) {
  std::ostringstream oss;
  oss << value;
  return oss.str();
}

std::string bool_string(const bool value) {
  return value ? "true" : "false";
}

std::string on_off_string(const bool value) {
  return value ? "on" : "off";
}

std::string join_strings(const std::vector<std::string>& values) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      oss << ",";
    }
    oss << values[i];
  }
  return oss.str();
}

std::string radiation_mode_string(const tenryu::core::RadiationMode mode) {
  switch (mode) {
    case tenryu::core::RadiationMode::ImcDdmc:
      return "imc_ddmc";
    case tenryu::core::RadiationMode::MultigroupDiffusion:
      return "multigroup_diffusion";
    case tenryu::core::RadiationMode::SnTransport:
      return "sn_transport";
  }
  return "unknown";
}

std::string summary_geometry(const tenryu::core::Config& cfg) {
  return (cfg.main.dimension == "2D_RZ") ? "rz" : cfg.mesh.geometry_1d;
}

std::string radiation_inner_boundary(const tenryu::core::Config& cfg) {
  switch (cfg.radiation.mode) {
    case tenryu::core::RadiationMode::MultigroupDiffusion:
      return cfg.radiation.multigroup_diffusion.boundary.inner_r;
    case tenryu::core::RadiationMode::SnTransport:
      return cfg.radiation.sn_transport.boundary.inner_r;
    case tenryu::core::RadiationMode::ImcDdmc:
      return cfg.radiation.boundary.inner_r;
  }
  return cfg.radiation.boundary.inner_r;
}

std::string radiation_outer_boundary(const tenryu::core::Config& cfg) {
  switch (cfg.radiation.mode) {
    case tenryu::core::RadiationMode::MultigroupDiffusion:
      return cfg.radiation.multigroup_diffusion.boundary.outer_r;
    case tenryu::core::RadiationMode::SnTransport:
      return cfg.radiation.sn_transport.boundary.outer_r;
    case tenryu::core::RadiationMode::ImcDdmc:
      return cfg.radiation.boundary.outer_r;
  }
  return cfg.radiation.boundary.outer_r;
}

std::string hot_electron_summary(const tenryu::core::Config& cfg) {
  if (!cfg.laser.hot_electron.enable) {
    return "off";
  }
  const std::size_t channels = cfg.laser.hot_electron.sources_specified
                                  ? cfg.laser.hot_electron.sources.size()
                                  : 1;
  return std::to_string(channels) + " channel(s)";
}

void log_preflight_summary(const tenryu::core::Config& cfg) {
  const int cell_count =
      (cfg.main.dimension == "2D_RZ") ? cfg.mesh.nr * cfg.mesh.nz : cfg.mesh.nr;

  tenryu::core::log_info("[TENRYU] ---- pre-flight summary ----");
  tenryu::core::log_info("[TENRYU] main      : name=" + cfg.main.name +
                         "  dimension=" + cfg.main.dimension +
                         "  geometry=" + summary_geometry(cfg) +
                         "  t_end=" + value_string(cfg.main.t_end) + " s");
  tenryu::core::log_info("[TENRYU] mesh      : nr=" + std::to_string(cfg.mesh.nr) +
                         (cfg.main.dimension == "2D_RZ"
                              ? " nz=" + std::to_string(cfg.mesh.nz)
                              : "") +
                         "  r=[" + value_string(cfg.mesh.r_min) + ", " +
                         value_string(cfg.mesh.r_max) + "] cm  cells=" +
                         std::to_string(cell_count));
  for (const auto& mat : cfg.materials.materials) {
    tenryu::core::log_info("[TENRYU] material  : " + mat.name +
                           "  A=" + value_string(mat.A) +
                           " Z=" + value_string(mat.Z) +
                           "  eos=" + mat.eos_model +
                           "  opacity=" + mat.opacity_model +
                           (mat.opacity_model == "constant"
                                ? " kappa_a=" + value_string(mat.kappa_a_constant) +
                                      " cm2/g"
                                : ""));
  }
  tenryu::core::log_info(
      "[TENRYU] radiation : enabled=" + bool_string(cfg.radiation.enabled) +
      "  mode=" + radiation_mode_string(cfg.radiation.mode) +
      "  groups=" + std::to_string(cfg.radiation.groups) +
      "  boundary inner=" + radiation_inner_boundary(cfg) +
      " outer=" + radiation_outer_boundary(cfg) +
      (cfg.radiation.boundary.marshak_Tr.detected ||
               !cfg.radiation.boundary.marshak_Tr_map.empty()
           ? " marshak_Tr=table"
           : ""));
  tenryu::core::log_info("[TENRYU] laser     : enabled=" +
                         bool_string(cfg.laser.enabled) +
                         (cfg.laser.enabled
                              ? "  mode=" + cfg.laser.mode +
                                    "  beams=" +
                                    std::to_string(cfg.laser.beams.size()) +
                                    "  cbet=" + on_off_string(cfg.laser.cbet.enable) +
                                    "  hot_electron=" + hot_electron_summary(cfg)
                              : ""));
  tenryu::core::log_info("[TENRYU] hydro     : enabled=" +
                         bool_string(cfg.numerics.hydro.enabled) +
                         (cfg.numerics.hydro.enabled
                              ? "  plasma_viscosity=" +
                                    on_off_string(
                                        cfg.numerics.hydro.plasma_viscosity.enabled)
                              : ""));
  tenryu::core::log_info("[TENRYU] conduction: enabled=" +
                         bool_string(cfg.numerics.conduction.enabled) +
                         (cfg.numerics.conduction.enabled
                              ? "  nonlocal=" +
                                    cfg.numerics.conduction.nonlocal_model
                              : ""));
  tenryu::core::log_info("[TENRYU] burn      : enabled=" +
                         bool_string(cfg.burn.enabled) +
                         (cfg.burn.enabled
                              ? "  scheme=" + cfg.burn.scheme +
                                    "  screening=" + cfg.burn.screening +
                                    "  fuels=" + join_strings(cfg.burn.fuels)
                              : ""));
  tenryu::core::log_info("[TENRYU] numerics  : dt_initial=" +
                         value_string(cfg.numerics.dt.initial_s) +
                         " s  dt_max=" +
                         value_string(cfg.numerics.dt.max_s) + " s");
  tenryu::core::log_info("[TENRYU] output    : dir=" + cfg.output.directory +
                         "  plot_every=" + std::to_string(cfg.output.plot_every) +
                         "  history_every=" +
                         std::to_string(cfg.output.history_every));
  tenryu::core::log_info("[TENRYU] ----------------------------");
}

#if TENRYU_ENABLE_PYTHON

namespace py = pybind11;

std::string safe_repr(const py::handle value) {
  try {
    return py::str(py::repr(value)).cast<std::string>();
  } catch (...) {
    return "<unrepresentable>";
  }
}

void log_callable_warning(const std::string& path, const std::string& detail) {
  tenryu::core::log_warning("[TENRYU][validate] callable '" + path + "' " + detail);
}

bool cast_numeric(const py::handle value, double* out) {
  if (py::isinstance<py::bool_>(value) || value.is_none()) {
    return false;
  }
  try {
    *out = py::cast<double>(value);
  } catch (...) {
    return false;
  }
  return true;
}

bool check_scalar_result(const std::string& path, const py::handle value) {
  double numeric = 0.0;
  if (!cast_numeric(value, &numeric)) {
    log_callable_warning(path, "returned non-numeric value: " + safe_repr(value));
    return false;
  }
  if (!std::isfinite(numeric)) {
    log_callable_warning(path, "returned non-finite value: " + safe_repr(value));
    return false;
  }
  return true;
}

bool check_velocity_result(const std::string& path,
                           const py::handle value,
                           const bool is_2d) {
  double numeric = 0.0;
  if (cast_numeric(value, &numeric)) {
    if (!std::isfinite(numeric)) {
      log_callable_warning(path, "returned non-finite value: " + safe_repr(value));
      return false;
    }
    return true;
  }

  if (!py::isinstance<py::sequence>(value) || py::isinstance<py::str>(value)) {
    log_callable_warning(path, "returned non-numeric value: " + safe_repr(value));
    return false;
  }

  const py::sequence seq = py::reinterpret_borrow<py::sequence>(value);
  if (seq.size() != 2) {
    log_callable_warning(path,
                         "returned velocity sequence with size=" +
                             std::to_string(seq.size()) +
                             " (expected 2): " + safe_repr(value));
    return false;
  }

  const std::size_t n_components_to_check = is_2d ? 2 : 1;
  for (std::size_t i = 0; i < n_components_to_check; ++i) {
    double component = 0.0;
    if (!cast_numeric(seq[i], &component) || !std::isfinite(component)) {
      log_callable_warning(path,
                           "returned non-finite/non-numeric velocity component[" +
                               std::to_string(i) + "]: " + safe_repr(seq[i]));
      return false;
    }
  }

  // 1D uses radial component only; 2nd component is ignored at runtime.
  return true;
}

int infer_spatial_nargs(const py::object& callable, const bool is_2d) {
  if (is_2d) {
    return 2;
  }
  try {
    const py::object code = callable.attr("__code__");
    const int argcount = py::cast<int>(code.attr("co_argcount"));
    return (argcount >= 2) ? 2 : 1;
  } catch (...) {
    return 1;
  }
}

py::object call_spatial(const py::object& callable,
                        const int nargs,
                        const double r,
                        const double z) {
  if (nargs <= 1) {
    return callable(r);
  }
  return callable(r, z);
}

void validate_time_callable(const tenryu::core::namelist::Builder& builder,
                            const std::string& path) {
  const auto it = builder.callable_objects.find(path);
  if (it == builder.callable_objects.end()) {
    return;
  }
  try {
    const py::object out = it->second(0.0);
    (void)check_scalar_result(path, out);
  } catch (const std::exception& e) {
    log_callable_warning(path, "raised exception at t=0.0: " + std::string(e.what()));
  }
}

void validate_spatial_callable(const tenryu::core::namelist::Builder& builder,
                               const std::string& path,
                               const bool is_2d,
                               const double r,
                               const double z,
                               const bool velocity) {
  const auto it = builder.callable_objects.find(path);
  if (it == builder.callable_objects.end()) {
    return;
  }

  try {
    const int nargs = infer_spatial_nargs(it->second, is_2d);
    const py::object out = call_spatial(it->second, nargs, r, z);
    if (velocity) {
      (void)check_velocity_result(path, out, is_2d);
    } else {
      (void)check_scalar_result(path, out);
    }
  } catch (const std::exception& e) {
    log_callable_warning(path,
                         "raised exception at representative coordinates (r=" +
                             std::to_string(r) + ", z=" + std::to_string(z) +
                             "): " + std::string(e.what()));
  }
}

void validate_callable_sanity(const tenryu::core::Config& cfg,
                              const tenryu::core::namelist::Builder& builder) {
  const bool is_2d = (cfg.main.dim == 2);

  const double r_sample =
      (std::isfinite(cfg.mesh.r_min) && std::isfinite(cfg.mesh.r_max) &&
       cfg.mesh.r_max > cfg.mesh.r_min)
          ? 0.5 * (cfg.mesh.r_min + cfg.mesh.r_max)
          : 0.0;
  const double z_sample =
      (is_2d && std::isfinite(cfg.mesh.z_min) && std::isfinite(cfg.mesh.z_max) &&
       cfg.mesh.z_max > cfg.mesh.z_min)
          ? 0.5 * (cfg.mesh.z_min + cfg.mesh.z_max)
          : 0.0;

  for (std::size_t i = 0; i < cfg.laser.beams.size(); ++i) {
    validate_time_callable(builder, "Laser.beams[" + std::to_string(i) + "].power");
  }
  if (cfg.radiation.boundary.marshak_Tr.detected) {
    validate_time_callable(builder, "Radiation.boundary.marshak_Tr");
  }
  for (const auto& [face, _] : cfg.radiation.boundary.marshak_Tr_map) {
    validate_time_callable(builder, "Radiation.boundary.marshak_Tr_map." + face);
  }
  if (cfg.numerics.hydro.pressure_drive_1d.detected) {
    validate_time_callable(builder, "Numerics.hydro.boundary_pressure");
  }

  validate_spatial_callable(builder, "Geometry.rho", is_2d, r_sample, z_sample, false);
  validate_spatial_callable(builder, "Geometry.Te", is_2d, r_sample, z_sample, false);
  validate_spatial_callable(builder, "Geometry.Ti", is_2d, r_sample, z_sample, false);
  if (cfg.geometry.velocity.detected) {
    validate_spatial_callable(builder, "Geometry.velocity", is_2d, r_sample, z_sample, true);
  }
}

#endif

void emit_mesh_preview_json(const tenryu::core::Config& cfg) {
  std::ostringstream oss;
  oss.setf(std::ios::fmtflags(0), std::ios::floatfield);
  oss << std::setprecision(17);
  const auto& mesh = cfg.mesh;
  const auto emit_nodes = [&oss](const std::vector<double>& nodes) {
    oss << "[";
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      if (i > 0) {
        oss << ",";
      }
      oss << nodes[i];
    }
    oss << "]";
  };
  oss << "{";
  oss << "\"dim\":" << cfg.main.dim;
  oss << ",\"dimension\":\"" << cfg.main.dimension << "\"";
  oss << ",\"logical_mesh_2d\":\"" << mesh.logical_mesh_2d << "\"";
  oss << ",\"geometry_1d\":\"" << mesh.geometry_1d << "\"";
  oss << ",\"nr\":" << mesh.nr;
  oss << ",\"nz\":" << mesh.nz;
  oss << ",\"r_min\":" << mesh.r_min;
  oss << ",\"r_max\":" << mesh.r_max;
  oss << ",\"z_min\":" << mesh.z_min;
  oss << ",\"z_max\":" << mesh.z_max;
  if (mesh.logical_mesh_2d != "rectangular_rz") {
    oss << ",\"polar\":{";
    oss << "\"s_max\":" << mesh.spherical_polar_s_max;
    oss << ",\"kappa\":" << mesh.spherical_polar_kappa;
    oss << ",\"center_treatment\":\"" << mesh.polar_center_treatment << "\"";
    oss << ",\"equal_mu\":" << (mesh.polar_equal_mu_zoning ? "true" : "false");
    oss << "}";
  } else {
    oss << ",\"polar\":null";
  }
  if (!mesh.explicit_nodes.empty()) {
    oss << ",\"r_nodes\":";
    emit_nodes(mesh.explicit_nodes);
  } else {
    oss << ",\"r_nodes\":null";
  }
  if (!mesh.explicit_nodes_z.empty()) {
    oss << ",\"z_nodes\":";
    emit_nodes(mesh.explicit_nodes_z);
  } else {
    oss << ",\"z_nodes\":null";
  }
  oss << "}";
  std::cout << "TENRYU-MESH-PREVIEW: " << oss.str() << std::endl;
}

}  // namespace

int cmd_validate(const std::string& namelist_path, const bool mesh_preview) {
#if TENRYU_ENABLE_PYTHON
  try {
    tenryu::core::namelist::PythonGuard python_guard;
    tenryu::core::namelist::Runtime runtime;
    runtime.execute(namelist_path);

    const auto& cfg = runtime.config();
    validate_callable_sanity(cfg, runtime.builder());
    tenryu::core::log_info("[TENRYU] Configuration validated successfully.");
    log_preflight_summary(cfg);
    if (mesh_preview) {
      emit_mesh_preview_json(cfg);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << format_cli_error(e.what()) << '\n';
    return 1;
  }
#else
  (void)namelist_path;
  tenryu::core::log_error("TENRYU was built without Python support (TENRYU_ENABLE_PYTHON=OFF)");
  return 1;
#endif
}

}  // namespace tenryu::drivers
