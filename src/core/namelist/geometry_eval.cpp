#include "core/namelist/geometry_eval.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/namelist/builder.hpp"
#include "core/namelist/geometry_eval_volume_cut.hpp"
#include "materials/ionmix_reader.hpp"
#include "materials/zbar_tf.hpp"

#if TENRYU_ENABLE_PYTHON
#include <pybind11/stl.h>
#endif

namespace tenryu::core::namelist {
namespace {

template <typename FieldLike>
ScalarRangeSummary summarize_scalar_field(const FieldLike& values) {
  ScalarRangeSummary summary;
  if (values.empty()) {
    return summary;
  }

  const auto [it_min, it_max] = std::minmax_element(values.begin(), values.end());
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);

  summary.min = *it_min;
  summary.max = *it_max;
  summary.mean = sum / static_cast<double>(values.size());
  summary.valid = true;
  return summary;
}

std::string cell_zero_volfrac_message(const std::size_t cell_id) {
  std::ostringstream oss;
  oss << "all volume fractions are zero in cell " << cell_id;
  return oss.str();
}

double sanitize_positive_callable_output(const char* field,
                                         const std::size_t cell_id,
                                         const double value) {
  if (!std::isfinite(value)) {
    std::ostringstream oss;
    oss << "Geometry callable produced non-finite " << field << " in cell " << cell_id
        << ": value=" << value;
    throw ConfigError(oss.str());
  }
  if (value >= 0.0) {
    return value;
  }

  std::ostringstream oss;
  oss << "Geometry callable produced negative " << field << " in cell " << cell_id
      << ": value=" << value << ", clamped=0";
  ::tenryu::core::log_warning(oss.str());
  return 0.0;
}

double sanitize_volfrac_callable_output(const std::size_t cell_id,
                                        const std::size_t material_index,
                                        const double value) {
  const bool invalid = !std::isfinite(value) || value < 0.0 || value > 1.0;
  if (!invalid) {
    return value;
  }

  double clamped = value;
  if (!std::isfinite(value)) {
    clamped = (value > 0.0) ? 1.0 : 0.0;
  } else {
    clamped = std::clamp(value, 0.0, 1.0);
  }

  std::ostringstream oss;
  oss << "Geometry callable produced invalid volfrac in cell " << cell_id
      << ", material " << material_index << ": value=" << value
      << ", clamped=" << clamped;
  ::tenryu::core::log_warning(oss.str());
  return clamped;
}

VolumeCutResult sample_button_center_region(
    const ::tenryu::mesh::Mesh& mesh,
    const std::vector<double>& host_x_r,
    const std::vector<double>& host_x_z,
    const std::size_t n_mat,
    const GeometryCallables& callables,
    const Config& cfg,
    const bool use_plic_t0_3x3) {
  TENRYU_ASSERT(mesh.button_center && mesh.button_center->enabled,
                "button center sampling requires button topology");
  const int c_button = mesh.topo.cell_index(0, 0);
  const double button_volume = mesh.cell_vol[static_cast<std::size_t>(c_button)];
  if (!(button_volume > 0.0)) {
    throw ConfigError("button center cell volume must be positive");
  }

  VolumeCutResult out;
  out.volfrac.assign(n_mat, 0.0);
  out.converged = true;
  out.total_volume = button_volume;

  double raw_volume = 0.0;
  double rho_int = 0.0;
  double Te_int = 0.0;
  double Ti_int = 0.0;
  std::vector<double> mat_volume(n_mat, 0.0);
  const int outer = mesh.button_center->outer_node_ring;
  for (int i = 0; i < outer; ++i) {
    for (int j = 0; j < mesh.topo.nz; ++j) {
      const int n00 = mesh.topo.node_index(i, j);
      const int n10 = mesh.topo.node_index(i + 1, j);
      const int n11 = mesh.topo.node_index(i + 1, j + 1);
      const int n01 = mesh.topo.node_index(i, j + 1);
      const VolumeCutResult cut = adaptive_volume_cut_sample_cell(
          host_x_r[static_cast<std::size_t>(n00)],
          host_x_z[static_cast<std::size_t>(n00)],
          host_x_r[static_cast<std::size_t>(n10)],
          host_x_z[static_cast<std::size_t>(n10)],
          host_x_r[static_cast<std::size_t>(n11)],
          host_x_z[static_cast<std::size_t>(n11)],
          host_x_r[static_cast<std::size_t>(n01)],
          host_x_z[static_cast<std::size_t>(n01)], n_mat, callables,
          cfg.numerics.plic.t0_volume_cut_max_depth,
          cfg.numerics.plic.t0_volume_cut_volfrac_tol, use_plic_t0_3x3);
      if (!(cut.total_volume > 0.0)) {
        continue;
      }
      raw_volume += cut.total_volume;
      rho_int += cut.rho_volume_avg * cut.total_volume;
      Te_int += cut.Te_volume_avg * cut.total_volume;
      Ti_int += cut.Ti_volume_avg * cut.total_volume;
      for (std::size_t m = 0; m < n_mat; ++m) {
        mat_volume[m] += cut.volfrac[m] * cut.total_volume;
      }
      out.converged = out.converged && cut.converged;
      out.max_depth_reached =
          std::max(out.max_depth_reached, cut.max_depth_reached);
      out.leaf_count += cut.leaf_count;
    }
  }
  if (!(raw_volume > 0.0)) {
    throw ConfigError("button center raw collapsed volume must be positive");
  }

  const double inv_v = 1.0 / button_volume;
  out.rho_volume_avg = rho_int * inv_v;
  out.Te_volume_avg = Te_int * inv_v;
  out.Ti_volume_avg = Ti_int * inv_v;
  for (std::size_t m = 0; m < n_mat; ++m) {
    out.volfrac[m] = mat_volume[m] * inv_v;
  }
  return out;
}

}  // namespace

GeometrySummary evaluate_geometry_from_callables(const Config& cfg,
                                                 State& state,
                                                 const GeometryCallables& callables) {
  const bool is_2d = (cfg.main.dim == 2);
  const std::size_t n_cells = static_cast<std::size_t>(state.mesh.topo.n_cells);
  const std::size_t n_nodes = static_cast<std::size_t>(state.mesh.topo.n_nodes);
  const std::size_t n_mat = cfg.materials.materials.size();

  TENRYU_ASSERT(state.vol.size() == n_cells,
                "State.vol size must match mesh cell count");
  TENRYU_ASSERT(state.rho.size() == n_cells,
                "State.rho size must match mesh cell count");
  TENRYU_ASSERT(state.Te.size() == n_cells,
                "State.Te size must match mesh cell count");
  TENRYU_ASSERT(state.Ti.size() == n_cells,
                "State.Ti size must match mesh cell count");
  TENRYU_ASSERT(state.mass.size() == n_cells,
                "State.mass size must match mesh cell count");
  TENRYU_ASSERT(state.zbar.size() == n_cells,
                "State.zbar size must match mesh cell count");
  TENRYU_ASSERT(state.volFrac.size() == n_cells * n_mat,
                "State.volFrac size must match n_cells * n_materials");
  TENRYU_ASSERT(state.mesh.cell_centroid_r.size() == n_cells,
                "Mesh centroid_r size must match n_cells");
  TENRYU_ASSERT(state.mesh.cell_centroid_z.size() == n_cells,
                "Mesh centroid_z size must match n_cells");
  TENRYU_ASSERT(state.x_r.size() == n_nodes,
                "State.x_r size must match mesh node count");
  TENRYU_ASSERT(state.x_z.size() == n_nodes,
                "State.x_z size must match mesh node count");
  TENRYU_ASSERT(state.v_r.size() == n_nodes,
                "State.v_r size must match mesh node count");
  TENRYU_ASSERT(state.v_z.size() == n_nodes,
                "State.v_z size must match mesh node count");
  TENRYU_ASSERT(state.cell_is_void.size() == n_cells,
                "State.cell_is_void size must match mesh cell count");

  if (callables.rho == nullptr || callables.Te == nullptr ||
      callables.Ti == nullptr) {
    throw ConfigError("Geometry callables rho/Te/Ti must be defined");
  }
  if (callables.volfrac.size() != n_mat) {
    throw ConfigError("Geometry volfrac callable count must match material count");
  }

  constexpr double kFracTol = 1.0e-12;
  std::vector<double> host_vol(n_cells, 0.0);
  std::vector<double> host_rho(n_cells, 0.0);
  std::vector<double> host_Te(n_cells, 0.0);
  std::vector<double> host_Ti(n_cells, 0.0);
  std::vector<double> host_mass(n_cells, 0.0);
  std::vector<double> host_zbar(n_cells, 0.0);
  std::vector<double> host_volFrac(n_cells * n_mat, 0.0);
  std::vector<double> host_x_r(n_nodes, 0.0);
  std::vector<double> host_x_z(n_nodes, 0.0);
  std::vector<double> host_v_r(n_nodes, 0.0);
  std::vector<double> host_v_z(n_nodes, 0.0);

  state.vol.copy_to_host(host_vol.data());
  state.x_r.copy_to_host(host_x_r.data());
  state.x_z.copy_to_host(host_x_z.data());

  int plic_t0_interface_cells = 0;
  const bool use_plic_t0_volume_cut =
      is_2d && cfg.numerics.plic.enabled &&
      cfg.numerics.plic.t0_volume_cut_method != "centroid_only_legacy";
  const bool use_plic_t0_3x3 =
      cfg.numerics.plic.t0_volume_cut_method == "adaptive_subdivision_3x3";

  for (std::size_t c = 0; c < n_cells; ++c) {
    const double r = state.mesh.cell_centroid_r[c];
    const double z = is_2d ? state.mesh.cell_centroid_z[c] : 0.0;

    const int c_int = static_cast<int>(c);
    if (state.mesh.is_dormant_cell(c_int)) {
      state.cell_is_void[c] = static_cast<std::uint8_t>(1);
      host_rho[c] = 0.0;
      host_Te[c] = 0.0;
      host_Ti[c] = 0.0;
      host_mass[c] = 0.0;
      host_zbar[c] = 0.0;
      continue;
    }

    double frac_sum = 0.0;
    if (state.mesh.is_button_cell(c_int)) {
      // Button IC volume-averages cell primitives only; node velocity and
      // EOS-derived energies remain initialized by their existing paths.
      const VolumeCutResult cut = sample_button_center_region(
          state.mesh, host_x_r, host_x_z, n_mat, callables, cfg,
          use_plic_t0_3x3);
      host_rho[c] = sanitize_positive_callable_output("rho", c, cut.rho_volume_avg);
      host_Te[c] = sanitize_positive_callable_output("Te", c, cut.Te_volume_avg);
      host_Ti[c] = sanitize_positive_callable_output("Ti", c, cut.Ti_volume_avg);
      for (std::size_t m = 0; m < n_mat; ++m) {
        const double value =
            sanitize_volfrac_callable_output(c, m, cut.volfrac[m]);
        const std::size_t idx = c * n_mat + m;
        host_volFrac[idx] = value;
        frac_sum += value;
      }
    } else if (use_plic_t0_volume_cut) {
      const int i = static_cast<int>(c) / cfg.mesh.nz;
      const int j = static_cast<int>(c) - i * cfg.mesh.nz;
      const int n00 = state.mesh.topo.node_index(i, j);
      const int n10 = state.mesh.topo.node_index(i + 1, j);
      const int n11 = state.mesh.topo.node_index(i + 1, j + 1);
      const int n01 = state.mesh.topo.node_index(i, j + 1);
      const VolumeCutResult cut = adaptive_volume_cut_sample_cell(
          host_x_r[static_cast<std::size_t>(n00)],
          host_x_z[static_cast<std::size_t>(n00)],
          host_x_r[static_cast<std::size_t>(n10)],
          host_x_z[static_cast<std::size_t>(n10)],
          host_x_r[static_cast<std::size_t>(n11)],
          host_x_z[static_cast<std::size_t>(n11)],
          host_x_r[static_cast<std::size_t>(n01)],
          host_x_z[static_cast<std::size_t>(n01)], n_mat, callables,
          cfg.numerics.plic.t0_volume_cut_max_depth,
          cfg.numerics.plic.t0_volume_cut_volfrac_tol, use_plic_t0_3x3);
      host_rho[c] = sanitize_positive_callable_output("rho", c, cut.rho_volume_avg);
      host_Te[c] = sanitize_positive_callable_output("Te", c, cut.Te_volume_avg);
      host_Ti[c] = sanitize_positive_callable_output("Ti", c, cut.Ti_volume_avg);
      for (std::size_t m = 0; m < n_mat; ++m) {
        const double value =
            sanitize_volfrac_callable_output(c, m, cut.volfrac[m]);
        const std::size_t idx = c * n_mat + m;
        host_volFrac[idx] = value;
        frac_sum += value;
      }
    } else {
      host_rho[c] = sanitize_positive_callable_output("rho", c, callables.rho(r, z));
      host_Te[c] = sanitize_positive_callable_output("Te", c, callables.Te(r, z));
      host_Ti[c] = sanitize_positive_callable_output("Ti", c, callables.Ti(r, z));

      for (std::size_t m = 0; m < n_mat; ++m) {
        const double value =
            sanitize_volfrac_callable_output(c, m, callables.volfrac[m](r, z));
        const std::size_t idx = c * n_mat + m;
        host_volFrac[idx] = value;
        frac_sum += value;
      }
    }

    if (frac_sum > 1.0 + kFracTol) {
      throw ConfigError("Geometry volume fractions exceed 1.0 in at least one cell");
    }

    if (cfg.geometry.enforce_sum_to_one) {
      if (frac_sum <= kFracTol) {
        throw ConfigError(cell_zero_volfrac_message(c));
      }
      for (std::size_t m = 0; m < n_mat; ++m) {
        const std::size_t idx = c * n_mat + m;
        host_volFrac[idx] /= frac_sum;
      }
      frac_sum = 1.0;
    } else if (frac_sum < 1.0 - kFracTol) {
      // Treat uncovered fraction as void by scaling effective density.
      host_rho[c] *= frac_sum;
    }

    if (use_plic_t0_volume_cut) {
      bool mixed = false;
      for (std::size_t m = 0; m < n_mat; ++m) {
        const double f = host_volFrac[c * n_mat + m];
        if (f > kFracTol && f < 1.0 - kFracTol) {
          mixed = true;
        }
      }
      if (mixed) {
        ++plic_t0_interface_cells;
      }
    }

    double nonvoid_sum = 0.0;
    for (std::size_t m = 0; m < n_mat; ++m) {
      if (!cfg.materials.materials[m].is_void) {
        nonvoid_sum += host_volFrac[c * n_mat + m];
      }
    }
    state.cell_is_void[c] =
        (nonvoid_sum <= kFracTol) ? static_cast<std::uint8_t>(1)
                                  : static_cast<std::uint8_t>(0);

    if (state.cell_is_void[c] != 0U) {
      host_rho[c] = cfg.materials.void_config.rho;
      host_Te[c] = cfg.materials.void_config.Te;
      host_Ti[c] = cfg.materials.void_config.Ti;
    }

    host_mass[c] = host_rho[c] * host_vol[c];

    if (cfg.materials.zbar.model == "fixed") {
      if (cfg.materials.zbar.fixed_value >= 0.0) {
        host_zbar[c] = cfg.materials.zbar.fixed_value;
      } else {
        double weighted = 0.0;
        double weight_sum = 0.0;
        for (std::size_t m = 0; m < n_mat; ++m) {
          if (cfg.materials.materials[m].is_void) {
            continue;
          }
          const double weight = host_volFrac[c * n_mat + m];
          weighted += weight * cfg.materials.materials[m].Z;
          weight_sum += weight;
        }
        host_zbar[c] = (weight_sum > kFracTol) ? (weighted / weight_sum) : 0.0;
      }
    } else if (cfg.materials.zbar.model == "thomas_fermi") {
      double weighted = 0.0;
      double weight_sum = 0.0;
      for (std::size_t m = 0; m < n_mat; ++m) {
        const auto& mat = cfg.materials.materials[m];
        if (mat.is_void) {
          continue;
        }
        const double weight = host_volFrac[c * n_mat + m];
        const double zbar_m =
            tenryu::materials::compute_zbar_tf(host_rho[c], host_Te[c], mat.Z, mat.A);
        weighted += weight * zbar_m;
        weight_sum += weight;
      }
      host_zbar[c] = (weight_sum > kFracTol) ? (weighted / weight_sum) : 0.0;
    } else if (cfg.materials.zbar.model == "tabular") {
      TENRYU_ASSERT(cfg.materials.zbar_tables.size() == n_mat,
                    "tabular Zbar tables not loaded");
      double weighted = 0.0;
      double weight_sum = 0.0;
      for (std::size_t m = 0; m < n_mat; ++m) {
        const auto& mat = cfg.materials.materials[m];
        if (mat.is_void) {
          continue;
        }
        TENRYU_ASSERT(cfg.materials.zbar_tables[m] != nullptr,
                      "tabular Zbar table missing for non-void material");
        const double weight = host_volFrac[c * n_mat + m];
        weighted += weight * cfg.materials.zbar_tables[m]->interpolate(host_rho[c], host_Te[c]);
        weight_sum += weight;
      }
      host_zbar[c] = (weight_sum > kFracTol) ? (weighted / weight_sum) : 0.0;
    } else {
      TENRYU_ASSERT(false, "Unknown materials.zbar.model");
    }

    if (state.cell_is_void[c] != 0U) {
      host_zbar[c] = 0.0;
    }
  }

  if (callables.velocity != nullptr) {
    for (std::size_t n = 0; n < n_nodes; ++n) {
      const double r = host_x_r[n];
      const double z = is_2d ? host_x_z[n] : 0.0;
      const auto v = callables.velocity(r, z);
      host_v_r[n] = v[0];
      host_v_z[n] = is_2d ? v[1] : 0.0;
    }
  }

  state.rho.copy_from_host(host_rho.data());
  state.Te.copy_from_host(host_Te.data());
  state.Ti.copy_from_host(host_Ti.data());
  state.mass.copy_from_host(host_mass.data());
  state.zbar.copy_from_host(host_zbar.data());
  state.volFrac.copy_from_host(host_volFrac.data());
  state.v_r.copy_from_host(host_v_r.data());
  state.v_z.copy_from_host(host_v_z.data());

  GeometrySummary summary;
  summary.rho = summarize_scalar_field(host_rho);
  summary.Te = summarize_scalar_field(host_Te);
  summary.Ti = summarize_scalar_field(host_Ti);
  summary.interface_cells_observed = plic_t0_interface_cells;

  for (std::size_t m = 0; m < n_mat; ++m) {
    double total = 0.0;
    for (std::size_t c = 0; c < n_cells; ++c) {
      total += host_volFrac[c * n_mat + m] * host_vol[c];
    }
    summary.material_volume[cfg.materials.materials[m].name] = total;
  }

  return summary;
}

#if TENRYU_ENABLE_PYTHON

namespace py = pybind11;

int infer_callable_args(const py::object& callable, const bool is_2d) {
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

const py::object& require_callable_object(const Builder& builder,
                                          const std::string& path) {
  const auto it = builder.callable_objects.find(path);
  if (it == builder.callable_objects.end()) {
    throw ConfigError("Missing callable object for " + path);
  }
  return it->second;
}

double eval_scalar_callable(const py::object& callable,
                            const int nargs,
                            const double r,
                            const double z) {
  if (nargs <= 1) {
    return py::cast<double>(callable(r));
  }
  return py::cast<double>(callable(r, z));
}

std::array<double, 2> eval_velocity_callable(const py::object& callable,
                                             const int nargs,
                                             const double r,
                                             const double z,
                                             const bool is_2d) {
  py::object out;
  if (nargs <= 1) {
    out = callable(r);
  } else {
    out = callable(r, z);
  }

  if (py::isinstance<py::sequence>(out)) {
    const py::sequence seq = py::reinterpret_borrow<py::sequence>(out);
    if (seq.size() != 2) {
      throw ConfigError("Geometry.velocity callable must return 2 components");
    }
    return {py::cast<double>(seq[0]), is_2d ? py::cast<double>(seq[1]) : 0.0};
  }

  return {py::cast<double>(out), 0.0};
}

GeometrySummary evaluate_geometry(const Config& cfg,
                                  const Builder& builder,
                                  State& state) {
  const bool is_2d = (cfg.main.dim == 2);

  const py::object& rho_obj = require_callable_object(builder, "Geometry.rho");
  const py::object& te_obj = require_callable_object(builder, "Geometry.Te");
  const py::object& ti_obj = require_callable_object(builder, "Geometry.Ti");

  const int rho_nargs = infer_callable_args(rho_obj, is_2d);
  const int te_nargs = infer_callable_args(te_obj, is_2d);
  const int ti_nargs = infer_callable_args(ti_obj, is_2d);

  GeometryCallables callables;
  callables.rho = [rho_obj, rho_nargs](const double r, const double z) {
    return eval_scalar_callable(rho_obj, rho_nargs, r, z);
  };
  callables.Te = [te_obj, te_nargs](const double r, const double z) {
    return eval_scalar_callable(te_obj, te_nargs, r, z);
  };
  callables.Ti = [ti_obj, ti_nargs](const double r, const double z) {
    return eval_scalar_callable(ti_obj, ti_nargs, r, z);
  };

  callables.volfrac.reserve(cfg.materials.materials.size());
  for (const auto& mat : cfg.materials.materials) {
    const std::string path = "Geometry.volfrac." + mat.name;
    const py::object& vf_obj = require_callable_object(builder, path);
    const int vf_nargs = infer_callable_args(vf_obj, is_2d);
    callables.volfrac.push_back(
        [vf_obj, vf_nargs](const double r, const double z) {
          return eval_scalar_callable(vf_obj, vf_nargs, r, z);
        });
  }

  const auto velocity_it = builder.callable_objects.find("Geometry.velocity");
  if (velocity_it != builder.callable_objects.end()) {
    const py::object velocity_obj = velocity_it->second;
    const int velocity_nargs = infer_callable_args(velocity_obj, is_2d);
    callables.velocity =
        [velocity_obj, velocity_nargs, is_2d](const double r, const double z) {
          return eval_velocity_callable(velocity_obj, velocity_nargs, r, z, is_2d);
        };
  }

  return evaluate_geometry_from_callables(cfg, state, callables);
}

#endif

void evaluate_geometry(const Config&, State&) {
  throw ConfigError("evaluate_geometry(cfg, state) requires callable context");
}

}  // namespace tenryu::core::namelist
