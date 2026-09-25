#include "core/mesh_requirement.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "materials/zbar_tf.hpp"

namespace tenryu::core {
namespace {

constexpr double kCriticalDensityCoefficient = 1.11485e21;
// Ion mass A * m_p and eV -> erg as in the solver (the laser map computes
// n_e / n_c with A_eff * proton_mass), so the requirement's critical density
// is the one the run sees.
constexpr double kIonMassUnitG = constants::proton_mass;
constexpr double kWattToErgPerSecond = 1.0e7;
constexpr double kEvToErg = constants::eV_to_erg;
constexpr double kDynPerMbar = 1.0e12;
constexpr int kSubintervalsPerPanel = 64;

struct RadialCell {
  double r_lo = 0.0;
  double r_hi = 0.0;
  double rho = 0.0;
  int material = -1;
  double depth_hi = 0.0;
  double depth_lo = 0.0;
};

struct RadialScan {
  std::vector<RadialCell> cells;
  int surface_cell = -1;
  double R0 = 0.0;
  double total_depth = 0.0;
  double total_nonvoid_depth = 0.0;
  double total_geometry_mass = 0.0;
};

[[nodiscard]] bool valid_material_index(const MeshRequirementInputs& in,
                                        const int material) {
  return material >= 0 &&
         material < static_cast<int>(in.materials.size());
}

[[nodiscard]] bool nonvoid_cell(const MeshRequirementInputs& in,
                                const RadialCell& cell) {
  return valid_material_index(in, cell.material) &&
         !in.materials[static_cast<std::size_t>(cell.material)].is_void &&
         cell.rho > in.rho_void_cut;
}

[[nodiscard]] double geometry_mass(const int geometry_code,
                                   const double rho,
                                   const double r_lo,
                                   const double r_hi) {
  if (geometry_code == 0) {
    return (4.0 * std::numbers::pi_v<double> / 3.0) * rho *
           (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
  }
  if (geometry_code == 1) {
    return std::numbers::pi_v<double> * rho *
           (r_hi * r_hi - r_lo * r_lo);
  }
  return rho * (r_hi - r_lo);
}

[[nodiscard]] RadialScan scan_radial_profile(
    const MeshRequirementInputs& in) {
  RadialScan scan;
  std::vector<double> panel_edges;
  panel_edges.reserve(in.breakpoints.size() + 2U);
  panel_edges.push_back(in.r_min);
  for (const double breakpoint : in.breakpoints) {
    if (breakpoint > in.r_min && breakpoint < in.r_max) {
      panel_edges.push_back(breakpoint);
    }
  }
  panel_edges.push_back(in.r_max);
  std::sort(panel_edges.begin(), panel_edges.end());
  panel_edges.erase(std::unique(panel_edges.begin(), panel_edges.end()),
                    panel_edges.end());

  scan.cells.reserve((panel_edges.size() - 1U) *
                     static_cast<std::size_t>(kSubintervalsPerPanel));
  for (std::size_t panel = 0; panel + 1U < panel_edges.size(); ++panel) {
    const double panel_lo = panel_edges[panel];
    const double panel_hi = panel_edges[panel + 1U];
    const double dr =
        (panel_hi - panel_lo) / static_cast<double>(kSubintervalsPerPanel);
    for (int k = 0; k < kSubintervalsPerPanel; ++k) {
      RadialCell cell;
      cell.r_lo = panel_lo + static_cast<double>(k) * dr;
      cell.r_hi = panel_lo + static_cast<double>(k + 1) * dr;
      const double r_mid = 0.5 * (cell.r_lo + cell.r_hi);
      cell.rho = in.rho0(r_mid);
      cell.material = in.material_at(r_mid);
      scan.cells.push_back(cell);
    }
  }

  for (std::size_t i = 0; i < scan.cells.size(); ++i) {
    if (nonvoid_cell(in, scan.cells[i])) {
      scan.surface_cell = static_cast<int>(i);
      scan.R0 = scan.cells[i].r_hi;
    }
  }
  if (scan.surface_cell < 0) {
    return scan;
  }

  double depth = 0.0;
  for (int i = scan.surface_cell; i >= 0; --i) {
    RadialCell& cell = scan.cells[static_cast<std::size_t>(i)];
    cell.depth_hi = depth;
    const double areal_mass = mesh_requirement_cell_areal_mass(
        in.geometry_code, scan.R0, cell.rho, cell.r_lo, cell.r_hi);
    depth += areal_mass;
    cell.depth_lo = depth;
    if (nonvoid_cell(in, cell)) {
      scan.total_nonvoid_depth += areal_mass;
    }
  }
  scan.total_depth = depth;

  for (const RadialCell& cell : scan.cells) {
    scan.total_geometry_mass +=
        geometry_mass(in.geometry_code, cell.rho, cell.r_lo, cell.r_hi);
  }
  return scan;
}

[[nodiscard]] const RadialCell* cell_at_depth(const RadialScan& scan,
                                              const double depth) {
  if (scan.surface_cell < 0) {
    return nullptr;
  }
  if (depth <= 0.0) {
    return &scan.cells[static_cast<std::size_t>(scan.surface_cell)];
  }
  for (int i = scan.surface_cell; i >= 0; --i) {
    const RadialCell& cell = scan.cells[static_cast<std::size_t>(i)];
    if (depth <= cell.depth_lo) {
      return &cell;
    }
  }
  return nullptr;
}

[[nodiscard]] int last_nonvoid_material(const MeshRequirementInputs& in,
                                        const RadialScan& scan) {
  for (int i = 0; i <= scan.surface_cell; ++i) {
    if (nonvoid_cell(in, scan.cells[static_cast<std::size_t>(i)])) {
      return scan.cells[static_cast<std::size_t>(i)].material;
    }
  }
  return -1;
}

[[nodiscard]] int front_material_at_depth(const MeshRequirementInputs& in,
                                          const RadialScan& scan,
                                          const double depth) {
  const RadialCell* cell = cell_at_depth(scan, depth);
  if (cell != nullptr && nonvoid_cell(in, *cell)) {
    return cell->material;
  }

  if (cell != nullptr) {
    const int start = static_cast<int>(cell - scan.cells.data());
    for (int i = start; i >= 0; --i) {
      if (nonvoid_cell(in, scan.cells[static_cast<std::size_t>(i)])) {
        return scan.cells[static_cast<std::size_t>(i)].material;
      }
    }
  }
  return last_nonvoid_material(in, scan);
}

[[nodiscard]] double radius_at_depth(const RadialScan& scan,
                                     const int geometry_code,
                                     const double depth) {
  if (scan.surface_cell < 0) {
    return 0.0;
  }
  if (depth <= 0.0) {
    return scan.R0;
  }
  const RadialCell* cell = cell_at_depth(scan, depth);
  if (cell == nullptr) {
    return scan.cells.front().r_lo;
  }
  if (cell->rho == 0.0) {
    return cell->r_hi;
  }

  const double remaining = depth - cell->depth_hi;
  if (geometry_code == 0) {
    const double radicand =
        cell->r_hi * cell->r_hi * cell->r_hi -
        3.0 * scan.R0 * scan.R0 * remaining / cell->rho;
    return std::cbrt(std::max(radicand, 0.0));
  }
  if (geometry_code == 1) {
    const double radicand = cell->r_hi * cell->r_hi -
                            2.0 * scan.R0 * remaining / cell->rho;
    return std::sqrt(std::max(radicand, 0.0));
  }
  return cell->r_hi - remaining / cell->rho;
}

[[nodiscard]] double geometry_mass_fraction_at(const RadialScan& scan,
                                               const int geometry_code,
                                               const double radius) {
  if (!(scan.total_geometry_mass > 0.0)) {
    return 0.0;
  }
  double mass = 0.0;
  for (const RadialCell& cell : scan.cells) {
    if (radius >= cell.r_hi) {
      mass += geometry_mass(geometry_code, cell.rho, cell.r_lo, cell.r_hi);
      continue;
    }
    if (radius > cell.r_lo) {
      mass += geometry_mass(geometry_code, cell.rho, cell.r_lo, radius);
    }
    break;
  }
  return mass / scan.total_geometry_mass;
}

[[nodiscard]] double interpolate(const std::vector<double>& x,
                                 const std::vector<double>& y,
                                 const double query) {
  if (x.empty() || y.empty()) {
    return 0.0;
  }
  if (query <= x.front()) {
    return y.front();
  }
  if (query >= x.back()) {
    return y.back();
  }
  const auto upper = std::lower_bound(x.begin(), x.end(), query);
  const std::size_t hi = static_cast<std::size_t>(upper - x.begin());
  const std::size_t lo = hi - 1U;
  const double dx = x[hi] - x[lo];
  if (!(dx > 0.0)) {
    return y[lo];
  }
  const double fraction = (query - x[lo]) / dx;
  return y[lo] + fraction * (y[hi] - y[lo]);
}

[[nodiscard]] double inverse_ablation_time(
    const MeshRequirementReport& report, const double depth) {
  if (depth <= 0.0) {
    return 0.0;
  }
  if (report.table_t_s.empty() || report.table_mu_abl_g_cm2.empty()) {
    return 0.0;
  }
  if (depth > report.table_mu_abl_g_cm2.back()) {
    return report.t_end_s;
  }
  const auto upper = std::lower_bound(report.table_mu_abl_g_cm2.begin(),
                                      report.table_mu_abl_g_cm2.end(), depth);
  const std::size_t hi =
      static_cast<std::size_t>(upper - report.table_mu_abl_g_cm2.begin());
  if (hi == 0U) {
    return report.table_t_s.front();
  }
  const std::size_t lo = hi - 1U;
  const double d0 = report.table_mu_abl_g_cm2[lo];
  const double d1 = report.table_mu_abl_g_cm2[hi];
  if (!(d1 > d0)) {
    return report.table_t_s[hi];
  }
  const double fraction = (depth - d0) / (d1 - d0);
  return report.table_t_s[lo] +
         fraction * (report.table_t_s[hi] - report.table_t_s[lo]);
}

[[nodiscard]] double critical_density_for_material(
    const MeshRequirementReport& report, const int material) {
  if (material < 0 || material >= static_cast<int>(report.materials.size())) {
    return 0.0;
  }
  return report.materials[static_cast<std::size_t>(material)].rho_c_gcc;
}

[[nodiscard]] double ceiling_at_depth(const MeshRequirementReport& report,
                                      const MeshRequirementInputs& in,
                                      const RadialScan& scan,
                                      const double depth) {
  const int material = front_material_at_depth(in, scan, depth);
  const double time =
      std::max(inverse_ablation_time(report, depth), report.t_formation_s);
  const double scale_length = interpolate(report.table_t_s, report.table_L_c_cm,
                                          time);
  return report.intensity_correction_factor *
         critical_density_for_material(report, material) * scale_length /
         static_cast<double>(report.params.zones_per_scale_length);
}

void append_layers(MeshRequirementReport& report,
                   const MeshRequirementInputs& in,
                   const RadialScan& scan) {
  int run_material = -1;
  double run_lo = 0.0;
  double run_hi = 0.0;
  for (int i = 0; i <= scan.surface_cell; ++i) {
    const RadialCell& cell = scan.cells[static_cast<std::size_t>(i)];
    if (!nonvoid_cell(in, cell)) {
      if (run_material >= 0) {
        report.layers.push_back({run_material, run_lo, run_hi});
        run_material = -1;
      }
      continue;
    }
    if (cell.material != run_material) {
      if (run_material >= 0) {
        report.layers.push_back({run_material, run_lo, run_hi});
      }
      run_material = cell.material;
      run_lo = cell.r_lo;
    }
    run_hi = cell.r_hi;
  }
  if (run_material >= 0) {
    report.layers.push_back({run_material, run_lo, run_hi});
  }
}

[[nodiscard]] std::string escaped_json_string(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      escaped.push_back('\\');
    }
    escaped.push_back(character);
  }
  return escaped;
}

void emit_string(std::ostringstream& out, const std::string& value) {
  out << '"' << escaped_json_string(value) << '"';
}

void emit_double(std::ostringstream& out, const double value) {
  if (std::isfinite(value)) {
    out << value;
  } else {
    out << "null";
  }
}

void emit_rule_check(std::ostringstream& out,
                     const MeshRequirementRuleCheck& rule) {
  out << "{\"applicable\":" << (rule.applicable ? "true" : "false")
      << ",\"n_checked\":" << rule.n_checked
      << ",\"n_violations\":" << rule.n_violations
      << ",\"max_ratio\":";
  emit_double(out, rule.max_ratio);
  out << ",\"worst\":{\"cell\":" << rule.worst_cell
      << ",\"r_lo_cm\":";
  emit_double(out, rule.worst_r_lo_cm);
  out << ",\"r_hi_cm\":";
  emit_double(out, rule.worst_r_hi_cm);
  out << ",\"areal_mass_g_cm2\":";
  emit_double(out, rule.worst_areal_mass_g_cm2);
  out << ",\"ceiling_g_cm2\":";
  emit_double(out, rule.worst_ceiling_g_cm2);
  out << ",\"ratio\":";
  emit_double(out, rule.max_ratio);
  out << "}}";
}

[[nodiscard]] std::string invalid_reason(const MeshRequirementInputs& in,
                                         const MeshRequirementParams& params) {
  if (!(in.r_max > in.r_min)) {
    return "invalid inputs: r_max <= r_min";
  }
  if (!(in.t_end > 0.0)) {
    return "invalid inputs: t_end <= 0";
  }
  if (!(in.wavelength_nm > 0.0)) {
    return "invalid inputs: wavelength_nm <= 0";
  }
  if (in.geometry_code < 0 || in.geometry_code > 2) {
    return "invalid inputs: geometry_code";
  }
  if (params.zones_per_scale_length < 1) {
    return "invalid inputs: zones_per_scale_length";
  }
  if (!(params.intensity_exponent >= 0.0)) {
    return "invalid inputs: intensity_exponent";
  }
  if (!(params.intensity_reference_W_cm2 > 0.0)) {
    return "invalid inputs: intensity_reference_W_cm2";
  }
  if (!(params.scale_length_factor > 0.0)) {
    return "invalid inputs: scale_length_factor";
  }
  if (!(params.ablation_mass_safety > 0.0)) {
    return "invalid inputs: ablation_mass_safety";
  }
  if (!(params.formation_ablated_fraction > 0.0 &&
        params.formation_ablated_fraction < 1.0)) {
    return "invalid inputs: formation_ablated_fraction";
  }
  if (!(params.absorbed_fraction > 0.0 && params.absorbed_fraction <= 1.0)) {
    return "invalid inputs: absorbed_fraction";
  }
  if (params.shock_cells_per_separation < 1) {
    return "invalid inputs: shock_cells_per_separation";
  }
  if (!(params.shock_event_min_separation_frac > 0.0 &&
        params.shock_event_min_separation_frac < 1.0)) {
    return "invalid inputs: shock_event_min_separation_frac";
  }
  if (params.min_cells_per_layer < 1) {
    return "invalid inputs: min_cells_per_layer";
  }
  if (params.n_bands < 1 || params.n_bands > 32) {
    return "invalid inputs: n_bands";
  }
  if (in.materials.empty()) {
    return "invalid inputs: materials empty";
  }
  if (!in.rho0) {
    return "invalid inputs: rho0 missing";
  }
  if (!in.material_at) {
    return "invalid inputs: material_at missing";
  }
  if (in.power_W.n_points < 2 ||
      in.power_W.n_points != static_cast<int>(in.power_W.x.size()) ||
      in.power_W.n_points != static_cast<int>(in.power_W.y.size())) {
    return "invalid inputs: power table";
  }
  for (int i = 0; i + 1 < in.power_W.n_points; ++i) {
    if (!(in.power_W.x[static_cast<std::size_t>(i + 1)] >
          in.power_W.x[static_cast<std::size_t>(i)])) {
      return "invalid inputs: power table";
    }
  }
  return {};
}

}  // namespace

double mesh_requirement_cell_areal_mass(const int geometry_code,
                                        const double R0_cm,
                                        const double rho,
                                        const double r_lo,
                                        const double r_hi) {
  if (geometry_code == 0) {
    TENRYU_ASSERT(R0_cm > 0.0,
                  "spherical reference radius must be positive");
    return rho * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo) /
           (3.0 * R0_cm * R0_cm);
  }
  if (geometry_code == 1) {
    TENRYU_ASSERT(R0_cm > 0.0,
                  "cylindrical reference radius must be positive");
    return rho * (r_hi * r_hi - r_lo * r_lo) / (2.0 * R0_cm);
  }
  TENRYU_ASSERT(geometry_code == 2,
                "mesh requirement geometry code must be 0, 1, or 2");
  return rho * (r_hi - r_lo);
}

MeshRequirementReport build_mesh_requirement(const MeshRequirementInputs& in,
                                             const MeshRequirementParams& params) {
  MeshRequirementReport report;
  report.params = params;
  report.wavelength_nm = in.wavelength_nm;
  report.geometry_code = in.geometry_code;
  report.r_min = in.r_min;
  report.r_max = in.r_max;
  report.t_end_s = in.t_end;
  report.n_samples = in.power_W.n_points;

  if (!params.enabled) {
    report.reason = "disabled_by_deck";
    return report;
  }
  if (!in.laser_enabled) {
    report.reason = "laser disabled";
    return report;
  }
  report.reason = invalid_reason(in, params);
  if (!report.reason.empty()) {
    return report;
  }

  std::vector<double> power_W(static_cast<std::size_t>(in.power_W.n_points), 0.0);
  double positive_power_integral = 0.0;
  for (int j = 0; j < in.power_W.n_points; ++j) {
    power_W[static_cast<std::size_t>(j)] =
        std::max(in.power_W.eval(in.power_W.x[static_cast<std::size_t>(j)]), 0.0);
    if (j > 0) {
      const std::size_t lo = static_cast<std::size_t>(j - 1);
      const std::size_t hi = static_cast<std::size_t>(j);
      positive_power_integral +=
          0.5 * (power_W[lo] + power_W[hi]) *
          (in.power_W.x[hi] - in.power_W.x[lo]);
    }
  }
  if (!(positive_power_integral > 0.0)) {
    report.reason = "no positive laser power";
    return report;
  }

  const RadialScan scan = scan_radial_profile(in);
  if (scan.surface_cell < 0) {
    report.reason = "no non-void material";
    return report;
  }
  report.R0_cm = scan.R0;
  if (in.geometry_code == 0) {
    report.area_cm2 = 4.0 * std::numbers::pi_v<double> * scan.R0 * scan.R0;
  } else if (in.geometry_code == 1) {
    report.area_cm2 = 2.0 * std::numbers::pi_v<double> * scan.R0;
  } else {
    report.area_cm2 = 1.0;
  }

  std::vector<double> intensity_W_cm2(power_W.size(), 0.0);
  for (std::size_t j = 0; j < power_W.size(); ++j) {
    intensity_W_cm2[j] = power_W[j] / report.area_cm2;
    report.peak_intensity_W_cm2 =
        std::max(report.peak_intensity_W_cm2, intensity_W_cm2[j]);
    if (j > 0U) {
      report.fluence_J_cm2 +=
          0.5 * (intensity_W_cm2[j - 1U] + intensity_W_cm2[j]) *
          (in.power_W.x[j] - in.power_W.x[j - 1U]);
    }
  }
  report.intensity_correction_factor = std::min(
      1.0, std::pow(report.peak_intensity_W_cm2 /
                        params.intensity_reference_W_cm2,
                    -params.intensity_exponent));

  const double wavelength_um = in.wavelength_nm * 1.0e-3;
  const double critical_number_density =
      kCriticalDensityCoefficient / (wavelength_um * wavelength_um);
  const double peak_intensity_erg =
      kWattToErgPerSecond * report.peak_intensity_W_cm2;
  report.materials.reserve(in.materials.size());
  for (const MeshRequirementMaterial& material : in.materials) {
    MeshRequirementMaterialInfo info{material.name, material.A, material.Z,
                                     0.0, 0.0, material.is_void, "void"};
    if (!material.is_void) {
      if (params.zbar_override > 0.0) {
        info.zbar = params.zbar_override;
        info.zbar_source = "override";
      } else if (material.Z <= 18.0) {
        info.zbar = material.Z;
        info.zbar_source = "full_ionization";
      } else {
        info.zbar = material.Z / 2.0;
        for (int iteration = 0; iteration < 5; ++iteration) {
          info.zbar = std::min(std::max(info.zbar, 1.0e-3), material.Z);
          const double rho_c = critical_number_density * kIonMassUnitG *
                               material.A / info.zbar;
          const double c_T = std::cbrt(params.absorbed_fraction *
                                       peak_intensity_erg / (4.0 * rho_c));
          const double temperature_eV =
              material.A * kIonMassUnitG * c_T * c_T /
              ((info.zbar + 1.0) * kEvToErg);
          info.zbar = tenryu::materials::compute_zbar_tf(
              rho_c, temperature_eV, material.Z, material.A);
        }
        info.zbar_source = "thomas_fermi";
      }
      info.zbar = std::min(std::max(info.zbar, 1.0e-3), material.Z);
      info.rho_c_gcc = critical_number_density * kIonMassUnitG * material.A /
                       info.zbar;
    }
    report.materials.push_back(std::move(info));
  }

  report.ablator_material =
      scan.cells[static_cast<std::size_t>(scan.surface_cell)].material;
  append_layers(report, in, scan);

  const std::size_t n_times = power_W.size();
  report.table_t_s = in.power_W.x;
  report.table_mu_abl_g_cm2.assign(n_times, 0.0);
  report.table_L_c_cm.assign(n_times, 0.0);
  std::vector<double> sound_speed(n_times, 0.0);
  std::vector<double> mass_rate(n_times, 0.0);
  for (std::size_t j = 0; j < n_times; ++j) {
    const double front_depth = j == 0U ? 0.0 :
                                            report.table_mu_abl_g_cm2[j - 1U];
    const int front_material = front_material_at_depth(in, scan, front_depth);
    const double rho_c = critical_density_for_material(report, front_material);
    if (intensity_W_cm2[j] > 0.0 && rho_c > 0.0) {
      sound_speed[j] = std::cbrt(
          params.absorbed_fraction * kWattToErgPerSecond *
          intensity_W_cm2[j] / (4.0 * rho_c));
      mass_rate[j] = rho_c * sound_speed[j];
    }
    if (j > 0U) {
      const double dt = report.table_t_s[j] - report.table_t_s[j - 1U];
      report.table_mu_abl_g_cm2[j] =
          report.table_mu_abl_g_cm2[j - 1U] +
          params.ablation_mass_safety * 0.5 *
              (mass_rate[j - 1U] + mass_rate[j]) * dt;
      const double scale_increment =
          params.scale_length_factor * 0.5 *
          (sound_speed[j - 1U] + sound_speed[j]) * dt;
      report.table_L_c_cm[j] = report.table_L_c_cm[j - 1U] + scale_increment;
      if (in.geometry_code == 0) {
        report.table_L_c_cm[j] =
            std::min(report.table_L_c_cm[j], scan.R0 / 2.0);
      }
    }
  }

  report.mu_abl_total_g_cm2 = report.table_mu_abl_g_cm2.back();
  if (scan.total_nonvoid_depth > 0.0) {
    report.ablated_mass_fraction = std::clamp(
        report.mu_abl_total_g_cm2 / scan.total_nonvoid_depth, 0.0, 1.0);
  }
  if (report.mu_abl_total_g_cm2 >= scan.total_nonvoid_depth) {
    report.notes.push_back(
        "Predicted ablated areal mass reaches or exceeds the non-void areal "
        "mass: the scaling-law model predicts burn-through; every non-void "
        "cell carries an ablation ceiling.");
  }
  report.depth_formation_g_cm2 =
      params.formation_ablated_fraction * report.mu_abl_total_g_cm2;
  report.t_formation_s =
      inverse_ablation_time(report, report.depth_formation_g_cm2);
  report.L_c_formation_cm = interpolate(
      report.table_t_s, report.table_L_c_cm, report.t_formation_s);
  report.ceiling_formation_g_cm2 =
      report.intensity_correction_factor *
      critical_density_for_material(report, report.ablator_material) *
      report.L_c_formation_cm /
      static_cast<double>(params.zones_per_scale_length);

  report.profile.reserve(33U);
  std::vector<double> profile_depths;
  profile_depths.reserve(33U);
  profile_depths.push_back(0.0);
  profile_depths.push_back(report.depth_formation_g_cm2);
  for (int k = 0; k < 31; ++k) {
    const double fraction = static_cast<double>(k) / 30.0;
    profile_depths.push_back(
        report.depth_formation_g_cm2 +
        fraction * (report.mu_abl_total_g_cm2 -
                    report.depth_formation_g_cm2));
  }
  for (const double depth : profile_depths) {
    const double time =
        std::max(inverse_ablation_time(report, depth), report.t_formation_s);
    const double scale_length =
        interpolate(report.table_t_s, report.table_L_c_cm, time);
    report.profile.push_back(
        {depth, time, scale_length, ceiling_at_depth(report, in, scan, depth)});
  }

  report.scale_length_track.reserve(101U);
  for (int k = 0; k <= 100; ++k) {
    const double time = in.t_end * static_cast<double>(k) / 100.0;
    report.scale_length_track.push_back(
        {time,
         interpolate(report.table_t_s, report.table_L_c_cm, time),
         interpolate(report.table_t_s, sound_speed, time),
         std::max(in.power_W.eval(time), 0.0) / report.area_cm2});
  }

  std::vector<double> ablation_pressure(power_W.size(), 0.0);
  double maximum_pressure = 0.0;
  for (std::size_t j = 0; j < power_W.size(); ++j) {
    if (intensity_W_cm2[j] > 0.0) {
      ablation_pressure[j] =
          40.0 * kDynPerMbar *
          std::pow(intensity_W_cm2[j] / 1.0e15 / wavelength_um,
                   2.0 / 3.0);
    }
    maximum_pressure = std::max(maximum_pressure, ablation_pressure[j]);
  }
  if (maximum_pressure > 0.0) {
    std::vector<MeshRequirementShockEvent> raw_events;
    for (int k = 0; k <= 6; ++k) {
      const double level = maximum_pressure * std::pow(2.0, -k);
      for (std::size_t j = 0; j < ablation_pressure.size(); ++j) {
        const double previous = j == 0U ? 0.0 : ablation_pressure[j - 1U];
        if (previous < level && level <= ablation_pressure[j]) {
          raw_events.push_back(
              {report.table_t_s[j], level / kDynPerMbar,
               intensity_W_cm2[j]});
          break;
        }
      }
    }
    std::sort(raw_events.begin(), raw_events.end(),
              [](const MeshRequirementShockEvent& lhs,
                 const MeshRequirementShockEvent& rhs) {
                if (lhs.t_s != rhs.t_s) {
                  return lhs.t_s < rhs.t_s;
                }
                return lhs.P_a_Mbar > rhs.P_a_Mbar;
              });
    const double merge_separation =
        params.shock_event_min_separation_frac * in.t_end;
    double previous_raw_time = 0.0;
    for (const MeshRequirementShockEvent& event : raw_events) {
      if (report.shock_events.empty() ||
          event.t_s - previous_raw_time >= merge_separation) {
        report.shock_events.push_back(event);
      } else {
        report.shock_events.back().P_a_Mbar =
            std::max(report.shock_events.back().P_a_Mbar, event.P_a_Mbar);
      }
      previous_raw_time = event.t_s;
    }
  }

  double maximum_nonvoid_density = 0.0;
  for (int i = 0; i <= scan.surface_cell; ++i) {
    const RadialCell& cell = scan.cells[static_cast<std::size_t>(i)];
    if (nonvoid_cell(in, cell)) {
      maximum_nonvoid_density = std::max(maximum_nonvoid_density, cell.rho);
      if (cell.depth_hi >= report.mu_abl_total_g_cm2) {
        report.rho_payload_gcc = std::max(report.rho_payload_gcc, cell.rho);
      }
    }
  }
  if (!(report.rho_payload_gcc > 0.0)) {
    report.rho_payload_gcc = maximum_nonvoid_density;
    report.notes.push_back(
        "No non-void material remains beyond the predicted ablation depth; "
        "rho_payload uses the maximum non-void initial density.");
  }

  report.shock_applicable = report.shock_events.size() >= 2U;
  if (report.shock_applicable) {
    report.shock_ceiling_g_cm2 = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k + 1U < report.shock_events.size(); ++k) {
      const MeshRequirementShockEvent& first = report.shock_events[k];
      const MeshRequirementShockEvent& second = report.shock_events[k + 1U];
      const double pressure_dyn = first.P_a_Mbar * kDynPerMbar;
      const double shock_speed = std::sqrt(
          (4.0 / 3.0) * pressure_dyn / report.rho_payload_gcc);
      const double delta_mu = report.rho_payload_gcc * shock_speed *
                              (second.t_s - first.t_s);
      report.shock_pairs.push_back(
          {first.t_s, second.t_s, shock_speed, delta_mu});
      report.shock_ceiling_g_cm2 = std::min(
          report.shock_ceiling_g_cm2,
          delta_mu / static_cast<double>(params.shock_cells_per_separation));
    }
    report.shock_ceiling_g_cm2 *= report.intensity_correction_factor;
  }

  auto append_band = [&](const std::string& kind, const double depth_lo,
                         const double depth_hi, const double ceiling) {
    const double r_hi = radius_at_depth(scan, in.geometry_code, depth_lo);
    const double r_lo = radius_at_depth(scan, in.geometry_code, depth_hi);
    double rho_max = 0.0;
    for (const RadialCell& cell : scan.cells) {
      if (cell.r_hi > r_lo && cell.r_lo < r_hi &&
          valid_material_index(in, cell.material) &&
          !in.materials[static_cast<std::size_t>(cell.material)].is_void) {
        rho_max = std::max(rho_max, cell.rho);
      }
    }
    report.bands_recommended.push_back(
        {kind, depth_lo, depth_hi, r_lo, r_hi, ceiling,
         geometry_mass_fraction_at(scan, in.geometry_code, r_lo),
         geometry_mass_fraction_at(scan, in.geometry_code, r_hi)});
    MeshRequirementBand& band = report.bands_recommended.back();
    band.rho_max_gcc = rho_max;
    band.width_max_cm =
        rho_max > 0.0 && std::isfinite(ceiling) && ceiling > 0.0
            ? ceiling / rho_max
            : std::numeric_limits<double>::infinity();
  };
  append_band("formation", 0.0, report.depth_formation_g_cm2,
              report.ceiling_formation_g_cm2);
  const double depth_limit = scan.total_depth;
  bool burn_through_note_added = false;
  for (int k = 1; k <= params.n_bands; ++k) {
    const double depth_lo =
        report.depth_formation_g_cm2 +
        (report.mu_abl_total_g_cm2 - report.depth_formation_g_cm2) *
            static_cast<double>(k - 1) / static_cast<double>(params.n_bands);
    double depth_hi =
        report.depth_formation_g_cm2 +
        (report.mu_abl_total_g_cm2 - report.depth_formation_g_cm2) *
            static_cast<double>(k) / static_cast<double>(params.n_bands);
    if (depth_lo >= depth_limit) {
      if (!burn_through_note_added) {
        report.notes.push_back(
            "Ablation bands beyond the target's total areal mass were "
            "skipped (burn-through predicted).");
        burn_through_note_added = true;
      }
      continue;
    }
    depth_hi = std::min(depth_hi, depth_limit);
    const double ceiling = ceiling_at_depth(report, in, scan, depth_lo);
    if (std::isfinite(ceiling) && ceiling > 0.0) {
      append_band("ablation", depth_lo, depth_hi, ceiling);
    } else {
      report.notes.push_back(
          "Skipped ablation band because its ceiling is not finite and "
          "positive.");
    }
  }
  if (!report.shock_applicable) {
    report.notes.push_back(
        "Skipped payload band because the shock criterion is not applicable.");
  } else if (!(report.mu_abl_total_g_cm2 < scan.total_nonvoid_depth)) {
    report.notes.push_back(
        "Skipped payload band because no non-void depth remains beyond the "
        "predicted ablation depth.");
  } else if (!(std::isfinite(report.shock_ceiling_g_cm2) &&
               report.shock_ceiling_g_cm2 > 0.0)) {
    report.notes.push_back(
        "Skipped payload band because its shock ceiling is not finite and "
        "positive.");
  } else {
    append_band("payload", report.mu_abl_total_g_cm2,
                scan.total_nonvoid_depth, report.shock_ceiling_g_cm2);
  }

  for (const MeshRequirementBand& band : report.bands_recommended) {
    if (std::isfinite(band.areal_mass_max_g_cm2) &&
        band.areal_mass_max_g_cm2 > 0.0 && band.rho_max_gcc > 0.0) {
      report.dr_min_admissible_cm =
          std::min(report.dr_min_admissible_cm, band.width_max_cm);
    }
  }

  report.applicable = true;
  report.reason.clear();
  return report;
}

MeshRequirementCheck check_mesh_requirement(
    const MeshRequirementReport& report, const MeshRequirementInputs& in,
    const std::vector<double>& nodes, const std::vector<double>& rho_cells,
    const std::vector<int>& material_cells) {
  TENRYU_ASSERT(!nodes.empty(),
                "mesh requirement check requires at least one node");
  const std::size_t n_cells = nodes.size() - 1U;
  TENRYU_ASSERT(rho_cells.size() == n_cells,
                "mesh requirement rho cell count must match nodes");
  TENRYU_ASSERT(material_cells.size() == n_cells,
                "mesh requirement material cell count must match nodes");
  for (std::size_t i = 0; i + 1U < nodes.size(); ++i) {
    TENRYU_ASSERT(nodes[i + 1U] > nodes[i],
                  "mesh requirement nodes must be strictly increasing");
  }

  MeshRequirementCheck result;
  result.n_cells = static_cast<int>(n_cells);
  if (report.applicable) {
    result.ablation.applicable = true;
    result.shock.applicable = report.shock_applicable;
  } else {
    result.ablation.applicable = false;
    result.shock.applicable = false;
  }

  std::vector<double> areal_mass;
  std::vector<double> depth;
  if (report.applicable) {
    areal_mass.resize(n_cells, 0.0);
    depth.resize(n_cells, 0.0);
    double outside_depth = 0.0;
    for (std::size_t reverse = n_cells; reverse > 0U; --reverse) {
      const std::size_t i = reverse - 1U;
      areal_mass[i] = mesh_requirement_cell_areal_mass(
          report.geometry_code, report.R0_cm, rho_cells[i], nodes[i],
          nodes[i + 1U]);
      depth[i] = outside_depth;
      outside_depth += areal_mass[i];
    }
  }

  auto is_nonvoid = [&](const std::size_t i) {
    const int material = material_cells[i];
    return material >= 0 &&
           material < static_cast<int>(in.materials.size()) &&
           !in.materials[static_cast<std::size_t>(material)].is_void &&
           rho_cells[i] > in.rho_void_cut;
  };
  auto update_worst = [&](MeshRequirementRuleCheck& rule, const std::size_t i,
                          const double cell_mass, const double ceiling) {
    const double ratio = ceiling > 0.0
                             ? cell_mass / ceiling
                             : std::numeric_limits<double>::infinity();
    const double tie_scale =
        std::max({1.0, std::abs(ratio), std::abs(rule.max_ratio)});
    if (ratio > rule.max_ratio || ratio == rule.max_ratio ||
        std::abs(ratio - rule.max_ratio) <= 1.0e-12 * tie_scale) {
      rule.max_ratio = ratio;
      rule.worst_cell = static_cast<int>(i);
      rule.worst_r_lo_cm = nodes[i];
      rule.worst_r_hi_cm = nodes[i + 1U];
      rule.worst_areal_mass_g_cm2 = cell_mass;
      rule.worst_ceiling_g_cm2 = ceiling;
    }
    if (ratio > 1.0 + 1.0e-12) {
      ++rule.n_violations;
    }
  };

  if (result.ablation.applicable) {
    for (std::size_t i = 0; i < n_cells; ++i) {
      if (!is_nonvoid(i) || !(depth[i] < report.mu_abl_total_g_cm2)) {
        continue;
      }
      ++result.ablation.n_checked;
      const double full_ablation_time =
          inverse_ablation_time(report, depth[i] + areal_mass[i]);
      const double time =
          std::max(full_ablation_time, report.t_formation_s);
      const double scale_length = interpolate(
          report.table_t_s, report.table_L_c_cm, time);
      const double ceiling =
          report.intensity_correction_factor *
          critical_density_for_material(report, material_cells[i]) *
          scale_length /
          static_cast<double>(report.params.zones_per_scale_length);
      update_worst(result.ablation, i, areal_mass[i], ceiling);
    }
  }

  if (result.shock.applicable) {
    for (std::size_t i = 0; i < n_cells; ++i) {
      if (!is_nonvoid(i) || !(depth[i] >= report.mu_abl_total_g_cm2)) {
        continue;
      }
      ++result.shock.n_checked;
      const double local_areal_mass =
          rho_cells[i] * (nodes[i + 1U] - nodes[i]);
      update_worst(result.shock, i, local_areal_mass,
                   report.shock_ceiling_g_cm2);
    }
  }

  int run_material = -1;
  int run_count = 0;
  auto finish_layer_run = [&]() {
    if (run_material >= 0 && run_count < report.params.min_cells_per_layer) {
      result.layer_violations.push_back(
          {run_material, run_count, report.params.min_cells_per_layer});
    }
    run_material = -1;
    run_count = 0;
  };
  for (std::size_t i = 0; i < n_cells; ++i) {
    if (!is_nonvoid(i)) {
      finish_layer_run();
      continue;
    }
    if (material_cells[i] != run_material) {
      finish_layer_run();
      run_material = material_cells[i];
    }
    ++run_count;
  }
  finish_layer_run();
  result.layers_ok = result.layer_violations.empty();
  result.ok = result.ablation.n_violations == 0 &&
              result.shock.n_violations == 0 && result.layers_ok;
  return result;
}

std::string mesh_requirement_json(const MeshRequirementReport& report,
                                  const MeshRequirementCheck* check) {
  std::ostringstream out;
  out.setf(std::ios::fmtflags(0), std::ios::floatfield);
  out << std::setprecision(17);
  out << "{\"schema\":\"tenryu.mesh_requirement.v1\""
      << ",\"applicable\":" << (report.applicable ? "true" : "false")
      << ",\"reason\":";
  emit_string(out, report.reason);
  out << ",\"params\":{\"enabled\":"
      << (report.params.enabled ? "true" : "false") << ",\"apply\":";
  emit_string(out, report.params.apply);
  out << ",\"zones_per_scale_length\":"
      << report.params.zones_per_scale_length
      << ",\"intensity_exponent\":";
  emit_double(out, report.params.intensity_exponent);
  out << ",\"intensity_reference_W_cm2\":";
  emit_double(out, report.params.intensity_reference_W_cm2);
  out << ",\"scale_length_factor\":";
  emit_double(out, report.params.scale_length_factor);
  out << ",\"ablation_mass_safety\":";
  emit_double(out, report.params.ablation_mass_safety);
  out << ",\"formation_ablated_fraction\":";
  emit_double(out, report.params.formation_ablated_fraction);
  out << ",\"absorbed_fraction\":";
  emit_double(out, report.params.absorbed_fraction);
  out << ",\"shock_cells_per_separation\":"
      << report.params.shock_cells_per_separation
      << ",\"shock_event_min_separation_frac\":";
  emit_double(out, report.params.shock_event_min_separation_frac);
  out << ",\"min_cells_per_layer\":" << report.params.min_cells_per_layer
      << ",\"zbar_override\":";
  emit_double(out, report.params.zbar_override);
  out << ",\"n_bands\":" << report.params.n_bands << '}';

  out << ",\"inputs\":{\"wavelength_nm\":";
  emit_double(out, report.wavelength_nm);
  out << ",\"geometry\":" << report.geometry_code << ",\"r_min\":";
  emit_double(out, report.r_min);
  out << ",\"r_max\":";
  emit_double(out, report.r_max);
  out << ",\"R0_cm\":";
  emit_double(out, report.R0_cm);
  out << ",\"area_cm2\":";
  emit_double(out, report.area_cm2);
  out << ",\"t_end_s\":";
  emit_double(out, report.t_end_s);
  out << ",\"n_samples\":" << report.n_samples
      << ",\"peak_intensity_W_cm2\":";
  emit_double(out, report.peak_intensity_W_cm2);
  out << ",\"fluence_J_cm2\":";
  emit_double(out, report.fluence_J_cm2);
  out << ",\"ablator\":";
  if (report.ablator_material >= 0 &&
      report.ablator_material < static_cast<int>(report.materials.size())) {
    const MeshRequirementMaterialInfo& ablator =
        report.materials[static_cast<std::size_t>(report.ablator_material)];
    out << "{\"name\":";
    emit_string(out, ablator.name);
    out << ",\"A\":";
    emit_double(out, ablator.A);
    out << ",\"Z\":";
    emit_double(out, ablator.Z);
    out << ",\"zbar\":";
    emit_double(out, ablator.zbar);
    out << ",\"zbar_source\":";
    emit_string(out, ablator.zbar_source);
    out << ",\"rho_c_gcc\":";
    emit_double(out, ablator.rho_c_gcc);
    out << '}';
  } else {
    out << "null";
  }
  out << ",\"materials\":[";
  for (std::size_t i = 0; i < report.materials.size(); ++i) {
    if (i > 0U) {
      out << ',';
    }
    const MeshRequirementMaterialInfo& material = report.materials[i];
    out << "{\"name\":";
    emit_string(out, material.name);
    out << ",\"A\":";
    emit_double(out, material.A);
    out << ",\"Z\":";
    emit_double(out, material.Z);
    out << ",\"is_void\":" << (material.is_void ? "true" : "false")
        << ",\"rho_c_gcc\":";
    emit_double(out, material.rho_c_gcc);
    out << '}';
  }
  out << "]}";

  out << ",\"ablation\":{\"intensity_correction_factor\":";
  emit_double(out, report.intensity_correction_factor);
  out << ",\"mu_abl_total_g_cm2\":";
  emit_double(out, report.mu_abl_total_g_cm2);
  out << ",\"ablated_mass_fraction\":";
  emit_double(out, report.ablated_mass_fraction);
  out << ",\"depth_formation_g_cm2\":";
  emit_double(out, report.depth_formation_g_cm2);
  out << ",\"t_formation_s\":";
  emit_double(out, report.t_formation_s);
  out << ",\"L_c_formation_cm\":";
  emit_double(out, report.L_c_formation_cm);
  out << ",\"ceiling_formation_g_cm2\":";
  emit_double(out, report.ceiling_formation_g_cm2);
  out << ",\"dr_min_admissible_cm\":";
  emit_double(out, report.dr_min_admissible_cm);
  out << ",\"profile\":[";
  for (std::size_t i = 0; i < report.profile.size(); ++i) {
    if (i > 0U) {
      out << ',';
    }
    const MeshRequirementProfilePoint& point = report.profile[i];
    out << "{\"depth_g_cm2\":";
    emit_double(out, point.depth_g_cm2);
    out << ",\"t_abl_s\":";
    emit_double(out, point.t_abl_s);
    out << ",\"L_c_cm\":";
    emit_double(out, point.L_c_cm);
    out << ",\"ceiling_g_cm2\":";
    emit_double(out, point.ceiling_g_cm2);
    out << '}';
  }
  out << "]}";

  out << ",\"scale_length_track\":[";
  for (std::size_t i = 0; i < report.scale_length_track.size(); ++i) {
    if (i > 0U) {
      out << ',';
    }
    const MeshRequirementTrackPoint& point = report.scale_length_track[i];
    out << "{\"t_s\":";
    emit_double(out, point.t_s);
    out << ",\"L_c_cm\":";
    emit_double(out, point.L_c_cm);
    out << ",\"c_T_cm_s\":";
    emit_double(out, point.c_T_cm_s);
    out << ",\"I_W_cm2\":";
    emit_double(out, point.I_W_cm2);
    out << '}';
  }
  out << ']';

  out << ",\"shocks\":{\"applicable\":"
      << (report.shock_applicable ? "true" : "false")
      << ",\"events\":[";
  for (std::size_t i = 0; i < report.shock_events.size(); ++i) {
    if (i > 0U) {
      out << ',';
    }
    const MeshRequirementShockEvent& event = report.shock_events[i];
    out << "{\"t_s\":";
    emit_double(out, event.t_s);
    out << ",\"P_a_Mbar\":";
    emit_double(out, event.P_a_Mbar);
    out << ",\"I_W_cm2\":";
    emit_double(out, event.I_W_cm2);
    out << '}';
  }
  out << "],\"pairs\":[";
  for (std::size_t i = 0; i < report.shock_pairs.size(); ++i) {
    if (i > 0U) {
      out << ',';
    }
    const MeshRequirementShockPair& pair = report.shock_pairs[i];
    out << "{\"t0_s\":";
    emit_double(out, pair.t0_s);
    out << ",\"t1_s\":";
    emit_double(out, pair.t1_s);
    out << ",\"u_s_cm_s\":";
    emit_double(out, pair.u_s_cm_s);
    out << ",\"delta_mu_g_cm2\":";
    emit_double(out, pair.delta_mu_g_cm2);
    out << '}';
  }
  out << "],\"ceiling_g_cm2\":";
  emit_double(out, report.shock_ceiling_g_cm2);
  out << ",\"rho_payload_gcc\":";
  emit_double(out, report.rho_payload_gcc);
  out << '}';

  out << ",\"layers\":[";
  for (std::size_t i = 0; i < report.layers.size(); ++i) {
    if (i > 0U) {
      out << ',';
    }
    const MeshRequirementLayer& layer = report.layers[i];
    out << "{\"material\":" << layer.material << ",\"r_lo_cm\":";
    emit_double(out, layer.r_lo_cm);
    out << ",\"r_hi_cm\":";
    emit_double(out, layer.r_hi_cm);
    out << '}';
  }
  out << ']';

  out << ",\"bands_recommended\":[";
  for (std::size_t i = 0; i < report.bands_recommended.size(); ++i) {
    if (i > 0U) {
      out << ',';
    }
    const MeshRequirementBand& band = report.bands_recommended[i];
    out << "{\"kind\":";
    emit_string(out, band.kind);
    out << ",\"depth_lo_g_cm2\":";
    emit_double(out, band.depth_lo_g_cm2);
    out << ",\"depth_hi_g_cm2\":";
    emit_double(out, band.depth_hi_g_cm2);
    out << ",\"r_lo_cm\":";
    emit_double(out, band.r_lo_cm);
    out << ",\"r_hi_cm\":";
    emit_double(out, band.r_hi_cm);
    out << ",\"areal_mass_max_g_cm2\":";
    emit_double(out, band.areal_mass_max_g_cm2);
    out << ",\"mass_frac_lo\":";
    emit_double(out, band.mass_frac_lo);
    out << ",\"mass_frac_hi\":";
    emit_double(out, band.mass_frac_hi);
    out << ",\"rho_max_gcc\":";
    emit_double(out, band.rho_max_gcc);
    out << ",\"width_max_cm\":";
    emit_double(out, band.width_max_cm);
    out << '}';
  }
  out << ']';

  out << ",\"provenance\":{\"model\":\"tenryu.mesh_requirement.v1\""
      << ",\"references\":["
      << "\"Scheiner&Schmitt 2021 doi:10.1063/5.0056006\","
      << "\"Regan+ 2007 doi:10.1063/1.2671690\","
      << "\"Lindl 1995 Phys.Plasmas 2,3933 (P_a coefficient)\"]"
      << ",\"calibration\":{"
      << "\"campaign\":\"mesh_convergence_20260903\","
      << "\"reference\":\"tools/assist/data/mesh_convergence_reference.json\","
      << "\"converged_cases\":22,"
      << "\"r_c_geometric_mean_before_calibration\":1.26,"
      << "\"r_c_min_before\":0.39,\"r_c_max_before\":5.28,"
      << "\"rule\":\"zones_per_scale_length 9; ceiling x min(1,(I_peak/1e14 W/cm2)^-0.4)\"}"
      << ",\"notes\":[";
  for (std::size_t i = 0; i < report.notes.size(); ++i) {
    if (i > 0U) {
      out << ',';
    }
    emit_string(out, report.notes[i]);
  }
  out << "]}";

  if (check != nullptr) {
    out << ",\"requirement_check\":{\"ok\":"
        << (check->ok ? "true" : "false")
        << ",\"n_cells\":" << check->n_cells << ",\"ablation\":";
    emit_rule_check(out, check->ablation);
    out << ",\"shock\":";
    emit_rule_check(out, check->shock);
    out << ",\"layers\":{\"ok\":"
        << (check->layers_ok ? "true" : "false")
        << ",\"violations\":[";
    for (std::size_t i = 0; i < check->layer_violations.size(); ++i) {
      if (i > 0U) {
        out << ',';
      }
      const MeshRequirementLayerViolation& violation =
          check->layer_violations[i];
      out << "{\"material\":" << violation.material
          << ",\"n_cells\":" << violation.n_cells
          << ",\"required\":" << violation.required << '}';
    }
    out << "]}}";
  }

  out << '}';
  return out.str();
}

}  // namespace tenryu::core
