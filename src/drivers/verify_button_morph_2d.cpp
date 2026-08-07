// Standing 2D RZ button-morph verification gate.
//
// Phase A freezes the S-C morph-null conservation and landing checks. Phase B
// freezes the S-D driven seam-crossing scar acceptance measurement.

#include "drivers/verify_button_morph_2d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/namelist/frozen_table.hpp"
#include "core/state.hpp"
#include "coupling/driver.hpp"
#include "diagnostics/energy_budget.hpp"
#include "drivers/verify_button_morph_mesh.hpp"
#include "hydro/button_morph_indexing.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::drivers {

namespace {

namespace frozen = verify_button_morph_mesh;

constexpr double kPhaseAPressure = 4.78948649422801392e11;
constexpr double kPhaseATStart = 0.5e-10;
constexpr double kPhaseATEnd = 1.5e-10;
constexpr double kPhaseARunEnd = 2.0e-10;
constexpr double kPhaseBPressure = 1.0e14;
constexpr double kPhaseBTStart = 1.89e-10;
constexpr double kPhaseBTEnd = 2.64e-10;
constexpr double kPhaseBRunEnd = 3.25e-10;
constexpr int kScarBins = 48;
constexpr double kPi = 3.14159265358979323846;

std::string button_morph_fmt(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(17) << value;
  return oss.str();
}

bool button_morph_report(const char* id,
                         const char* name,
                         const bool pass,
                         const std::string& details) {
  const std::string line =
      std::string("[verify:button_morph_2d] ") + id + " " + name + " " +
      (pass ? "PASS" : "FAIL") + (details.empty() ? "" : " " + details);
  if (pass) {
    core::log_info(line);
  } else {
    core::log_error(line);
  }
  return pass;
}

bool button_morph_cuda_available() {
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count <= 0) {
    core::log_info("[SKIP] button_morph_2d: CUDA not available");
    return false;
  }
  return true;
}

core::namelist::FrozenTable1D constant_pressure_table(
    const double pressure,
    const double t_end) {
  core::namelist::FrozenTable1D table;
  table.x = {0.0, t_end};
  table.y = {pressure, pressure};
  table.n_points = 2;
  table.x_min = table.x.front();
  table.x_max = table.x.back();
  table.zero_outside = false;
  return table;
}

core::Config make_button_morph_config(const char* name,
                                      const double morph_t_start,
                                      const double morph_t_end,
                                      const double run_t_end,
                                      const bool equal_split_all) {
  core::Config cfg;
  cfg.main.name = name;
  cfg.main.dimension = "2D_RZ";
  cfg.main.temperature_model = "1T";
  cfg.main.two_temperature = false;
  cfg.main.dim = 2;
  cfg.main.t_end = run_t_end;

  cfg.mesh.nr = frozen::kNr;
  cfg.mesh.nz = frozen::kNz;
  cfg.mesh.r_min = frozen::kRMinCm;
  cfg.mesh.r_max = frozen::kRMaxCm;
  cfg.mesh.z_min = frozen::kZMinCm;
  cfg.mesh.z_max = frozen::kZMaxCm;
  cfg.mesh.grid_type_r = "uniform";
  cfg.mesh.grid_type_z = "uniform";
  cfg.mesh.motion = "ale";
  cfg.mesh.logical_mesh_2d = "spherical_polar_halfplane";
  cfg.mesh.spherical_polar_s_max = frozen::kMaxRadiusCm;
  cfg.mesh.topology_scheme =
      core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL;
  cfg.mesh.topology_scheme_explicit = true;
  cfg.mesh.multiblock_cart_core_n_c = frozen::kCoreCells;
  cfg.mesh.multiblock_cart_core_bridge_layers = frozen::kBridgeLayers;
  cfg.mesh.multiblock_cart_core_r_c = frozen::kCoreRadiusCm;
  cfg.mesh.multiblock_cart_core_r_match = frozen::kMatchRadiusCm;
  cfg.mesh.multiblock_cart_core_bridge_grading = "quintic_log";
  // BUG-25 opt-out (deck-faithful): the default-True outer-shell Svec tangent balance
  // deletes tangential restoring forces and kills driven button runs at the outer pole.
  cfg.mesh.multiblock_outer_svec_tangent_balance = false;
  cfg.mesh.explicit_nodes.assign(frozen::kShellNodes,
                                 frozen::kShellNodes + frozen::kNr + 1);

  core::Config::MaterialsConfig::MatDef material;
  material.name = "gas";
  material.A = frozen::kMaterialA;
  material.Z = frozen::kMaterialZ;
  material.eos_model = "ideal_gas";
  material.ideal_gas_gamma = frozen::kIdealGasGamma;
  material.opacity_model = "constant";
  material.kappa_a_constant = frozen::kOpacityCm2PerG;
  cfg.materials.materials = {material};

  cfg.numerics.dt.initial_s = frozen::kDtInitialS;
  cfg.numerics.dt.max_s = frozen::kDtMaxS;
  cfg.numerics.dt.cfl_hydro = frozen::kHydroCfl;
  // qualification deck value; the default 1.2 ramps the drive onset too
  // coarsely (phase-B outer-pole inversion at 19 ps) and amplifies the phase-A
  // equilibration ringing.
  cfg.numerics.dt.growth_factor = 1.02;
  cfg.numerics.floors.rho = frozen::kRhoFloorGPerCc;
  cfg.numerics.floors.Te = frozen::kTeFloorEv;
  cfg.numerics.floors.Ti = frozen::kTiFloorEv;
  cfg.numerics.has_physical_rz_axis = true;

  auto& boundary = cfg.numerics.hydro.boundary_2d;
  boundary.r_inner = "axis";
  boundary.r_outer = "pressure";
  boundary.z_bottom_cfg.type = "reflect";
  boundary.z_top_cfg.type = "reflect";
  boundary.sync_legacy_strings();
  if (equal_split_all) {
    cfg.numerics.hydro.axis_node_mass_convention = "equal_split_all";
  }

  cfg.numerics.ale.enabled = true;
  cfg.numerics.ale.every_n_steps = 1000000000;
  cfg.numerics.ale.button_morph.enabled = true;
  cfg.numerics.ale.button_morph.t_start_s = morph_t_start;
  cfg.numerics.ale.button_morph.t_end_s = morph_t_end;
  cfg.numerics.ale.button_morph.max_step_fraction = 0.05;
  cfg.numerics.ale.button_morph.every_n_steps = 1;

  cfg.radiation.enabled = false;
  cfg.numerics.conduction.enabled = false;
  cfg.burn.enabled = false;
  cfg.laser.enabled = false;

  cfg.diagnostics.enabled = false;
  cfg.diagnostics.energy_budget.enabled = false;
  cfg.diagnostics.areal_density.enabled = false;
  cfg.diagnostics.sphericity.enabled = false;
  cfg.diagnostics.laser_pattern.enabled = false;
  cfg.diagnostics.mc_stats.enabled = false;
  cfg.diagnostics.fleck_diag.enabled = false;
  cfg.diagnostics.overshoot_monitor = false;
  cfg.numerics.diagnostics.dt_breakdown_history_enabled = false;
  cfg.numerics.diagnostics.shock_approach.enabled = false;

  return cfg;
}

template <typename Tag>
std::vector<double> button_morph_to_host(const core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  if (!host.empty()) {
    field.copy_to_host(host.data());
  }
  return host;
}

core::State make_button_morph_state(core::Config& cfg,
                                    const double pressure) {
  core::State state = core::State::allocate(cfg);
  state.mesh = mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;
  state.pressure_drive_1d = constant_pressure_table(pressure, cfg.main.t_end);

  const std::vector<double> volume = button_morph_to_host(state.vol);
  std::vector<double> rho(state.rho.size(), frozen::kInitialRhoGPerCc);
  std::vector<double> mass(state.mass.size(), 0.0);
  std::vector<double> temperature(state.Te.size(),
                                  frozen::kInitialTemperatureEv);
  for (std::size_t cell = 0; cell < mass.size(); ++cell) {
    mass[cell] = rho[cell] * volume[cell];
  }
  state.rho.copy_from_host(rho);
  state.mass.copy_from_host(mass);
  state.Te.copy_from_host(temperature);
  state.Ti.copy_from_host(temperature);
  state.zbar.fill(frozen::kMaterialZ);
  state.volFrac.fill(1.0);
  state.v_r.fill(0.0);
  state.v_z.fill(0.0);

  coupling::initialize_eos_fields_if_needed(state, cfg);
  return state;
}

long double total_mass(const core::State& state) {
  const std::vector<double> mass = button_morph_to_host(state.mass);
  long double total = 0.0L;
  for (const double value : mass) {
    total += static_cast<long double>(value);
  }
  return total;
}

double total_energy(const core::State& state) {
  const diagnostics::EnergyTotals totals =
      diagnostics::compute_energy_totals_2d(state);
  return totals.E_int_e + totals.E_int_i + totals.E_kin;
}

struct PhaseAResult {
  bool returned = false;
  double t = std::numeric_limits<double>::quiet_NaN();
  double mass_rel = std::numeric_limits<double>::quiet_NaN();
  double energy_rel = std::numeric_limits<double>::quiet_NaN();
  double corner_rel = std::numeric_limits<double>::quiet_NaN();
  double shell_max_displacement = std::numeric_limits<double>::quiet_NaN();
};

PhaseAResult run_phase_a() {
  PhaseAResult result;
  try {
    core::Config cfg = make_button_morph_config(
        "verify_button_morph_2d_phase_a", kPhaseATStart, kPhaseATEnd,
        kPhaseARunEnd, false);
    core::State state = make_button_morph_state(cfg, kPhaseAPressure);
    const hydro::button_morph::ButtonIndexing idx =
        hydro::button_morph::make_button_indexing(cfg.mesh);

    const long double mass_0 = total_mass(state);
    const double energy_0 = total_energy(state);
    const std::vector<double> r_0 = button_morph_to_host(state.x_r);
    const std::vector<double> z_0 = button_morph_to_host(state.x_z);

    coupling::Driver driver;
    driver.run(state, cfg);
    result.returned = true;
    result.t = state.t;

    const long double mass_end = total_mass(state);
    const double energy_end = total_energy(state);
    result.mass_rel =
        static_cast<double>(std::fabs(mass_end / mass_0 - 1.0L));
    result.energy_rel = std::fabs(energy_end / energy_0 - 1.0);

    const std::vector<double> r_end = button_morph_to_host(state.x_r);
    const std::vector<double> z_end = button_morph_to_host(state.x_z);
    const int corner_node = hydro::button_morph::core_node_id(
        idx, idx.n_c, 2 * idx.n_c);
    const std::size_t corner = static_cast<std::size_t>(corner_node);
    const double a = std::max(r_0[corner], std::fabs(z_0[corner]));
    const double expected_radius = std::cbrt(1.5) * a;
    result.corner_rel =
        std::fabs(std::hypot(r_end[corner], z_end[corner]) /
                      expected_radius -
                  1.0);

    result.shell_max_displacement = 0.0;
    for (int node = idx.shell_node_offset; node < idx.n_nodes_total; ++node) {
      const std::size_t n = static_cast<std::size_t>(node);
      result.shell_max_displacement =
          std::max(result.shell_max_displacement,
                   std::hypot(r_end[n] - r_0[n], z_end[n] - z_0[n]));
    }
  } catch (const std::exception& error) {
    core::log_error(std::string("[verify:button_morph_2d] phase A exception: ") +
                    error.what());
  } catch (...) {
    core::log_error("[verify:button_morph_2d] phase A unknown exception");
  }
  return result;
}

struct ScarMetrics {
  double rms = std::numeric_limits<double>::quiet_NaN();
  double max_abs = std::numeric_limits<double>::quiet_NaN();
  int nonempty_bins = 0;
};

ScarMetrics compute_scar_metrics(const core::State& state) {
  ScarMetrics metrics;
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "button_morph_2d scar metric requires multiblock topology");
  const mesh::MultiBlockTopology& topology = *state.mesh.topo.multiblock;
  const std::vector<double> rho = button_morph_to_host(state.rho);
  const std::vector<double> r = button_morph_to_host(state.x_r);
  const std::vector<double> z = button_morph_to_host(state.x_z);
  std::array<long double, kScarBins> bin_sum{};
  std::array<int, kScarBins> bin_count{};

  TENRYU_ASSERT(topology.cell_node_csr_offsets.size() == rho.size() + 1,
                "button_morph_2d scar metric CSR offset size mismatch");
  for (std::size_t cell = 0; cell < rho.size(); ++cell) {
    const int begin = topology.cell_node_csr_offsets[cell];
    const int end = topology.cell_node_csr_offsets[cell + 1];
    TENRYU_ASSERT(end > begin,
                  "button_morph_2d scar metric requires non-empty CSR cells");
    long double centroid_r = 0.0L;
    long double centroid_z = 0.0L;
    for (int slot = begin; slot < end; ++slot) {
      const int node =
          topology.cell_node_csr_indices[static_cast<std::size_t>(slot)];
      centroid_r += r[static_cast<std::size_t>(node)];
      centroid_z += z[static_cast<std::size_t>(node)];
    }
    const long double inv_count =
        1.0L / static_cast<long double>(end - begin);
    const double cr = static_cast<double>(centroid_r * inv_count);
    const double cz = static_cast<double>(centroid_z * inv_count);
    const double radius = std::hypot(cr, cz);
    if (!(radius > 10.0e-4 && radius < 20.0e-4)) {
      continue;
    }
    const double theta = std::atan2(cr, cz);
    int bin = static_cast<int>(theta * static_cast<double>(kScarBins) / kPi);
    bin = std::clamp(bin, 0, kScarBins - 1);
    bin_sum[static_cast<std::size_t>(bin)] += rho[cell];
    ++bin_count[static_cast<std::size_t>(bin)];
  }

  std::array<double, kScarBins> bin_mean{};
  long double mean_sum = 0.0L;
  for (int bin = 0; bin < kScarBins; ++bin) {
    const int count = bin_count[static_cast<std::size_t>(bin)];
    if (count == 0) {
      continue;
    }
    bin_mean[static_cast<std::size_t>(bin)] = static_cast<double>(
        bin_sum[static_cast<std::size_t>(bin)] /
        static_cast<long double>(count));
    mean_sum += bin_mean[static_cast<std::size_t>(bin)];
    ++metrics.nonempty_bins;
  }
  if (metrics.nonempty_bins == 0) {
    return metrics;
  }

  const double mean = static_cast<double>(
      mean_sum / static_cast<long double>(metrics.nonempty_bins));
  if (!(mean > 0.0)) {
    return metrics;
  }
  long double square_sum = 0.0L;
  metrics.max_abs = 0.0;
  for (int bin = 0; bin < kScarBins; ++bin) {
    if (bin_count[static_cast<std::size_t>(bin)] == 0) {
      continue;
    }
    const double deviation =
        (bin_mean[static_cast<std::size_t>(bin)] - mean) / mean;
    square_sum += static_cast<long double>(deviation) * deviation;
    metrics.max_abs = std::max(metrics.max_abs, std::fabs(deviation));
  }
  metrics.rms = std::sqrt(static_cast<double>(
      square_sum / static_cast<long double>(metrics.nonempty_bins)));
  return metrics;
}

struct PhaseBResult {
  bool returned = false;
  double t = std::numeric_limits<double>::quiet_NaN();
  ScarMetrics scar;
};

PhaseBResult run_phase_b() {
  PhaseBResult result;
  try {
    core::Config cfg = make_button_morph_config(
        "verify_button_morph_2d_phase_b", kPhaseBTStart, kPhaseBTEnd,
        kPhaseBRunEnd, true);
    core::State state = make_button_morph_state(cfg, kPhaseBPressure);
    coupling::Driver driver;
    driver.run(state, cfg);
    result.returned = true;
    result.t = state.t;
    result.scar = compute_scar_metrics(state);
  } catch (const std::exception& error) {
    core::log_error(std::string("[verify:button_morph_2d] phase B exception: ") +
                    error.what());
  } catch (...) {
    core::log_error("[verify:button_morph_2d] phase B unknown exception");
  }
  return result;
}

}  // namespace

int verify_button_morph_2d() {
  if (!button_morph_cuda_available()) {
    return 0;
  }

  const PhaseAResult phase_a = run_phase_a();
  bool pass = true;
  pass &= button_morph_report(
      "A1", "run_reached",
      phase_a.returned && phase_a.t >= 1.999e-10,
      "t=" + button_morph_fmt(phase_a.t) + " threshold=1.99900000000000000e-10");
  pass &= button_morph_report(
      "A2", "mass_conservation",
      phase_a.returned && phase_a.mass_rel < 1.0e-12,
      "relative_error=" + button_morph_fmt(phase_a.mass_rel) +
          " threshold=1.00000000000000000e-12");
  pass &= button_morph_report(
      "A3", "energy_conservation",
      phase_a.returned && phase_a.energy_rel < 1.0e-12,
      "relative_error=" + button_morph_fmt(phase_a.energy_rel) +
          " threshold=1.00000000000000000e-12");
  pass &= button_morph_report(
      "A4", "corner_landing",
      // The time-bounded window ends with the last transaction at
      // alpha = 1 - O((dt/T)^3) (quintic end phase), so the corner lands within
      // ~2e-8 relative of the Shirley-Chiu radius; exact alpha=1 landing is
      // pinned by the host twins. 5e-7 covers any step phase.
      phase_a.returned && phase_a.corner_rel < 5.0e-7,
      "relative_error=" + button_morph_fmt(phase_a.corner_rel) +
          " threshold=5.00000000000000000e-07");
  pass &= button_morph_report(
      "A5", "shell_immobility",
      phase_a.returned && phase_a.shell_max_displacement < 1.0e-5,
      "max_displacement_cm=" +
          button_morph_fmt(phase_a.shell_max_displacement) +
          " threshold_cm=1.00000000000000000e-05");

  const PhaseBResult phase_b = run_phase_b();
  pass &= button_morph_report(
      "B1", "run_reached",
      phase_b.returned && phase_b.t >= 3.245e-10,
      "t=" + button_morph_fmt(phase_b.t) + " threshold=3.24500000000000000e-10");
  pass &= button_morph_report(
      "B2", "seam_crossing_scar",
      phase_b.returned && phase_b.scar.rms <= 0.015 &&
          phase_b.scar.max_abs <= 0.035,
      "rms=" + button_morph_fmt(phase_b.scar.rms) +
          " rms_threshold=1.50000000000000000e-02 max_abs=" +
          button_morph_fmt(phase_b.scar.max_abs) +
          " max_threshold=3.50000000000000000e-02 nonempty_bins=" +
          std::to_string(phase_b.scar.nonempty_bins));

  core::log_info(std::string("[verify:button_morph_2d] ") +
                 (pass ? "PASSED" : "FAILED"));
  return pass ? 0 : 1;
}

}  // namespace tenryu::drivers
