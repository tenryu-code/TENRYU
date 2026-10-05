// Reference outputs of the solver's 1D zoning (src/core/zoning_intent.cpp) for
// test/zoningIntent.test.ts, which checks Studio's TypeScript copy (src/core/deck/zoningIntent.ts)
// against them: compute_zoning_intent_nodes on fixed intents covering every input (pins with and
// without a ratio jump, profile, anchors, bands, the global cell-measure box, dr_min, the four
// measures, piecewise-constant and exponential densities) and the main failure codes. Prints the
// inputs and the results as JSON with 17 significant digits. Build and run from the repository
// root with a C++20 compiler (the solver's platform, Linux and glibc, for the stored file):
//   g++ -std=c++20 -O2 -Isrc gui/scripts/zoningIntentGolden.cpp src/core/zoning_intent.cpp -o zoning_golden
//   ./zoning_golden > gui/test/fixtures/zoningIntentGolden.json
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/zoning_intent.hpp"

namespace {

using tenryu::core::ZoningIntentConfig;
using tenryu::core::ZoningMeasure;

struct DensityRegion {
  double r_end;
  double rho;
};

// rho0 of a case: none (width), piecewise constant over regions (as the namelist builder makes it
// from zoning_intent.density_regions), or a constant core with an exponential fall to a floor.
struct Density {
  std::string kind = "none";
  std::vector<DensityRegion> regions;
  double rho_inner = 0.0;
  double r_start = 0.0;
  double scale = 0.0;
  double rho_floor = 0.0;
};

struct Case {
  std::string name;
  double r_min;
  double r_max;
  Density density;
  ZoningIntentConfig cfg;
};

std::function<double(double)> make_rho0(const Density& density) {
  if (density.kind == "regions") {
    std::vector<double> r_ends;
    std::vector<double> densities;
    for (const DensityRegion& region : density.regions) {
      r_ends.push_back(region.r_end);
      densities.push_back(region.rho);
    }
    return [r_ends, densities](const double r) {
      std::size_t index = static_cast<std::size_t>(
          std::upper_bound(r_ends.begin(), r_ends.end(), r) - r_ends.begin());
      index = std::min(index, densities.size() - 1);
      return densities[index];
    };
  }
  if (density.kind == "exponential") {
    return [density](const double r) {
      if (r < density.r_start) return density.rho_inner;
      return std::max(density.rho_inner * std::exp(-(r - density.r_start) / density.scale),
                      density.rho_floor);
    };
  }
  return {};
}

std::string measure_name(const ZoningMeasure measure) {
  switch (measure) {
    case ZoningMeasure::kWidth:
      return "width";
    case ZoningMeasure::kArealMass:
      return "areal_mass";
    case ZoningMeasure::kCylindricalLineMass:
      return "cylindrical_line_mass";
    case ZoningMeasure::kSphericalCellMass:
      return "spherical_cell_mass";
  }
  return "";
}

std::string status_name(const tenryu::core::ZoningStatus status) {
  switch (status) {
    case tenryu::core::ZoningStatus::kOk:
      return "ok";
    case tenryu::core::ZoningStatus::kInvalidInput:
      return "invalid_input";
    case tenryu::core::ZoningStatus::kInfeasible:
      return "infeasible";
    case tenryu::core::ZoningStatus::kNumericalFailure:
      return "numerical_failure";
  }
  return "";
}

std::string num(const double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

std::string text(const std::string& value) {
  std::string out = "\"";
  for (const char c : value) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

std::string numbers(const std::vector<double>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) out += (i > 0 ? "," : "") + num(values[i]);
  return out + "]";
}

void print_case(const Case& c, const bool last) {
  const ZoningIntentConfig& cfg = c.cfg;
  std::cout << "    {\n      \"name\": " << text(c.name) << ",\n";
  std::cout << "      \"rMin\": " << num(c.r_min) << ", \"rMax\": " << num(c.r_max) << ",\n";
  std::cout << "      \"density\": {\"kind\": " << text(c.density.kind);
  if (c.density.kind == "regions") {
    std::cout << ", \"regions\": [";
    for (std::size_t i = 0; i < c.density.regions.size(); ++i) {
      std::cout << (i > 0 ? ", " : "") << "{\"rEnd\": " << num(c.density.regions[i].r_end)
                << ", \"rho\": " << num(c.density.regions[i].rho) << "}";
    }
    std::cout << "]";
  } else if (c.density.kind == "exponential") {
    std::cout << ", \"rhoInner\": " << num(c.density.rho_inner) << ", \"rStart\": "
              << num(c.density.r_start) << ", \"scale\": " << num(c.density.scale)
              << ", \"rhoFloor\": " << num(c.density.rho_floor);
  }
  std::cout << "},\n      \"cfg\": {\"nCells\": " << cfg.n_cells << ", \"measure\": "
            << text(measure_name(cfg.measure)) << ",\n        \"pins\": [";
  for (std::size_t i = 0; i < cfg.pins.size(); ++i) {
    std::cout << (i > 0 ? ", " : "") << "{\"r\": " << num(cfg.pins[i].r)
              << ", \"ratioJumpAllowed\": " << (cfg.pins[i].ratio_jump_allowed ? "true" : "false")
              << "}";
  }
  std::cout << "],\n        \"profile\": [";
  for (std::size_t i = 0; i < cfg.profile.size(); ++i) {
    std::cout << (i > 0 ? ", " : "") << "{\"r\": " << num(cfg.profile[i].r) << ", \"w\": "
              << num(cfg.profile[i].w) << "}";
  }
  std::cout << "],\n        \"anchors\": [";
  for (std::size_t i = 0; i < cfg.anchors.size(); ++i) {
    std::cout << (i > 0 ? ", " : "") << "{\"r\": " << num(cfg.anchors[i].r)
              << ", \"halfWidth\": " << num(cfg.anchors[i].half_width)
              << ", \"logAmplitude\": " << num(cfg.anchors[i].log_amplitude) << "}";
  }
  std::cout << "],\n        \"bands\": [";
  for (std::size_t i = 0; i < cfg.bands.size(); ++i) {
    std::cout << (i > 0 ? ", " : "") << "{\"measureFracBegin\": "
              << num(cfg.bands[i].measure_frac_begin) << ", \"measureFracEnd\": "
              << num(cfg.bands[i].measure_frac_end) << ", \"cellMeasureMin\": "
              << num(cfg.bands[i].cell_measure_min) << ", \"cellMeasureMax\": "
              << num(cfg.bands[i].cell_measure_max) << "}";
  }
  std::cout << "],\n        \"extraEvents\": " << numbers(cfg.extra_events)
            << ", \"drMin\": " << num(cfg.dr_min) << ", \"cellMeasureMin\": "
            << num(cfg.cell_measure_min) << ", \"cellMeasureMax\": " << num(cfg.cell_measure_max)
            << ",\n        \"preferredRatio\": " << num(cfg.preferred_ratio)
            << ", \"ratioHardMax\": " << num(cfg.ratio_hard_max)
            << ", \"minCellsPerSegment\": " << cfg.min_cells_per_segment << "},\n";

  const tenryu::core::ZoningResult result =
      tenryu::core::compute_zoning_intent_nodes(c.r_min, c.r_max, cfg, make_rho0(c.density));
  const auto& d = result.diag;
  std::cout << "      \"result\": {\"ok\": " << (result.ok ? "true" : "false") << ", \"status\": "
            << text(status_name(d.status)) << ", \"code\": " << text(d.code)
            << ",\n        \"message\": " << text(d.message) << ",\n        \"warnings\": [";
  for (std::size_t i = 0; i < d.warnings.size(); ++i) {
    std::cout << (i > 0 ? ", " : "") << text(d.warnings[i]);
  }
  std::cout << "],\n        \"cellsPerSegment\": [";
  for (std::size_t i = 0; i < d.cells_per_segment.size(); ++i) {
    std::cout << (i > 0 ? ", " : "") << d.cells_per_segment[i];
  }
  std::cout << "], \"ratioMaxAchieved\": " << num(d.ratio_max_achieved)
            << ", \"nRatioSoftExceed\": " << d.n_ratio_soft_exceed
            << ",\n        \"widthMinAchieved\": " << num(d.width_min_achieved)
            << ", \"cellMeasureMinAchieved\": " << num(d.cell_measure_min_achieved)
            << ", \"cellMeasureMaxAchieved\": " << num(d.cell_measure_max_achieved)
            << ",\n        \"nodes\": " << numbers(result.nodes) << "}\n    }"
            << (last ? "" : ",") << "\n";
}

ZoningIntentConfig base(const int n_cells, const ZoningMeasure measure) {
  ZoningIntentConfig cfg;
  cfg.n_cells = n_cells;
  cfg.measure = measure;
  return cfg;
}

Density regions(std::vector<DensityRegion> list) {
  Density density;
  density.kind = "regions";
  density.regions = std::move(list);
  return density;
}

// Region ends strictly inside the domain as quadrature events (the namelist builder adds them).
void add_region_events(Case& c) {
  for (const DensityRegion& region : c.density.regions) {
    if (region.r_end > c.r_min && region.r_end < c.r_max) c.cfg.extra_events.push_back(region.r_end);
  }
}

std::vector<Case> cases() {
  std::vector<Case> out;
  {
    Case c{"width_uniform", 0.0, 0.01, {}, base(40, ZoningMeasure::kWidth)};
    out.push_back(c);
  }
  {
    Case c{"width_profile_pins_dr_min", 0.001, 0.05, {}, base(120, ZoningMeasure::kWidth)};
    c.cfg.pins = {{0.02, false}, {0.035, true}};
    c.cfg.profile = {{0.001, 1.0}, {0.02, 0.4}, {0.03, 0.1}, {0.05, 1.0}};
    c.cfg.dr_min = 5.0e-5;
    c.cfg.preferred_ratio = 1.25;
    c.cfg.ratio_hard_max = 1.5;
    c.cfg.min_cells_per_segment = 5;
    out.push_back(c);
  }
  {
    Case c{"areal_planar_layers_band", 0.0, 0.0125,
           regions({{0.004, 1.05}, {0.006, 2.7}, {0.0125, 0.05}}),
           base(200, ZoningMeasure::kArealMass)};
    add_region_events(c);
    c.cfg.pins = {{0.004, true}, {0.006, true}};
    c.cfg.profile = {{0.0, 3.0}, {0.006, 1.0}, {0.0125, 0.3}};
    c.cfg.bands = {{0.9, 1.0, 0.0, 2.0e-5}};
    c.cfg.min_cells_per_segment = 10;
    out.push_back(c);
  }
  for (const bool tight : {false, true}) {
    // The tight bands need more shell cells than the allocation gives the shell segment.
    Case c{tight ? "error_chain_sum_bands" : "spherical_shell_bands", 0.0, 0.03,
           regions({{0.023, 0.0025}, {0.025, 1.05}, {0.03, 1.0e-4}}),
           base(300, ZoningMeasure::kSphericalCellMass)};
    add_region_events(c);
    c.cfg.pins = {{0.023, true}, {0.025, true}};
    c.cfg.profile = {{0.0, 1.0}, {0.023, 0.5}, {0.025, 0.05}, {0.03, 0.05}};
    c.cfg.bands = tight ? std::vector<tenryu::core::ZoningBand>{{0.5, 0.9, 0.0, 4.0e-8}, {0.9, 1.0, 0.0, 1.5e-8}}
                        : std::vector<tenryu::core::ZoningBand>{{0.5, 0.9, 0.0, 1.0e-7}, {0.9, 1.0, 0.0, 4.0e-8}};
    c.cfg.dr_min = 1.0e-7;
    c.cfg.min_cells_per_segment = 20;
    out.push_back(c);
  }
  {
    Case c{"cylindrical_anchors_exponential", 0.0, 0.02, {}, base(150, ZoningMeasure::kCylindricalLineMass)};
    c.density.kind = "exponential";
    c.density.rho_inner = 1.0;
    c.density.r_start = 0.01;
    c.density.scale = 0.001;
    c.density.rho_floor = 1.0e-3;
    c.cfg.extra_events = {0.01, 0.01 + 0.001 * std::log(1.0e3)};
    c.cfg.anchors = {{0.012, 0.002, -1.5}, {0.005, 0.003, 0.7}};
    out.push_back(c);
  }
  {
    Case c{"areal_global_box", 0.0, 0.01, regions({{0.01, 1.0}}), base(80, ZoningMeasure::kArealMass)};
    c.cfg.profile = {{0.0, 1.0}, {0.01, 10.0}};
    c.cfg.cell_measure_min = 5.0e-5;
    c.cfg.cell_measure_max = 2.0e-4;
    out.push_back(c);
  }
  {
    Case c{"error_pin_out_of_domain", 0.0, 1.0, {}, base(20, ZoningMeasure::kWidth)};
    c.cfg.pins = {{1.0, false}};
    out.push_back(c);
  }
  {
    Case c{"error_segment_min_count", 0.0, 1.0, {}, base(20, ZoningMeasure::kWidth)};
    c.cfg.pins = {{0.3, true}, {0.6, true}};
    c.cfg.min_cells_per_segment = 10;
    out.push_back(c);
  }
  {
    Case c{"error_dr_min_capacity", 0.0, 1.0, {}, base(20, ZoningMeasure::kWidth)};
    c.cfg.dr_min = 0.1;
    out.push_back(c);
  }
  {
    Case c{"error_undeclared_jump", 0.0, 1.0, regions({{0.5, 1.0}, {1.0, 100.0}}),
           base(50, ZoningMeasure::kArealMass)};
    out.push_back(c);
  }
  {
    Case c{"error_cell_measure_box", 0.0, 1.0, regions({{1.0, 1.0}}), base(50, ZoningMeasure::kArealMass)};
    c.cfg.cell_measure_max = 0.01;
    out.push_back(c);
  }
  {
    Case c{"error_ratio_across_pin", 0.0, 1.0, regions({{1.0, 1.0}}), base(100, ZoningMeasure::kArealMass)};
    c.cfg.pins = {{0.9, false}};
    c.cfg.min_cells_per_segment = 40;
    out.push_back(c);
  }
  return out;
}

}  // namespace

int main() {
  const std::vector<Case> all = cases();
  std::cout << "{\n  \"generator\": \"gui/scripts/zoningIntentGolden.cpp\",\n  \"cases\": [\n";
  for (std::size_t i = 0; i < all.size(); ++i) print_case(all[i], i + 1 == all.size());
  std::cout << "  ]\n}\n";
  return 0;
}
