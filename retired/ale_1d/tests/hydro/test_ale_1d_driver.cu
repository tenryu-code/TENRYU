#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/state.hpp"
#include "coupling/driver.hpp"
#include "hydro/ale_1d_driver.cuh"
#include "hydro/eos_context.hpp"
#include "materials/eos_table.hpp"
#include "mesh/mesh.hpp"

namespace {

constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

template <typename Tag>
std::vector<double> to_host(const tenryu::core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

tenryu::core::Config make_cfg(const int n) {
  tenryu::core::Config cfg;
  cfg.main.dim = 1;
  cfg.main.dimension = "1D_SPH";
  // make_state builds a 2T state (separate e_e and e_i). The Config default
  // is 1T, whose closure folds e_i into e_e; the ALE's own reclosure used to
  // close with the 2T formulas either way, so these cases ran a 2T state in
  // a 1T configuration unnoticed.
  cfg.main.two_temperature = true;
  cfg.mesh.nr = n;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = 1.0;
  cfg.mesh.grid_type_r = "uniform";
  cfg.radiation.groups = 1;
  cfg.radiation.enabled = false;
  cfg.laser.enabled = false;
  cfg.numerics.conduction.enabled = false;
  cfg.numerics.ale1d.enabled = true;
  cfg.numerics.ale1d.min_cells = 8;
  cfg.numerics.ale1d.min_movable_segment_hard = 2;
  cfg.numerics.ale1d.min_movable_segment_warn = 2;
  cfg.numerics.ale1d.enable_benefit_gate = false;
  cfg.numerics.ale1d.laser_sensor.enabled = false;
  cfg.numerics.ale1d.ablation_sensor.enabled = false;
  cfg.numerics.ale1d.shock_sensor.enabled = false;
  cfg.numerics.ale1d.interface_sensor.enabled = false;
  cfg.numerics.ale1d.center_sensor.enabled = false;

  tenryu::core::Config::MaterialsConfig::MatDef mat;
  mat.name = "fuel";
  mat.A = 1.0;
  mat.Z = 1.0;
  mat.ideal_gas_gamma = 5.0 / 3.0;
  cfg.materials.materials = {mat};
  return cfg;
}

void enable_center_sensor(tenryu::core::Config& cfg) {
  cfg.numerics.ale1d.center_sensor.enabled = true;
}

double cv_mass(const tenryu::core::Config& cfg) {
  const auto& mat = cfg.materials.materials.front();
  return kEvToErg / (mat.A * kProtonMass * (mat.ideal_gas_gamma - 1.0));
}

tenryu::core::State make_state(const tenryu::core::Config& cfg) {
  tenryu::core::State state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;

  const int n = cfg.mesh.nr;
  const std::vector<double> vol = to_host(state.vol);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
  std::vector<double> zbar(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ee(static_cast<std::size_t>(n), cv_mass(cfg) * 1.0e-3);
  std::vector<double> ei(static_cast<std::size_t>(n), cv_mass(cfg) * 1.0e-3);
  std::vector<double> Te(static_cast<std::size_t>(n), 1.0e-3);
  std::vector<double> Ti(static_cast<std::size_t>(n), 1.0e-3);
  std::vector<double> Pe(static_cast<std::size_t>(n), 0.0);
  std::vector<double> Pi(static_cast<std::size_t>(n), 0.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 0.5);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vr(static_cast<std::size_t>(n + 1), 0.0);
  const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
  for (int i = 0; i < n; ++i) {
    mass[static_cast<std::size_t>(i)] = rho[static_cast<std::size_t>(i)] *
                                        vol[static_cast<std::size_t>(i)];
    Pe[static_cast<std::size_t>(i)] =
        (gamma - 1.0) * rho[static_cast<std::size_t>(i)] *
        ee[static_cast<std::size_t>(i)];
    Pi[static_cast<std::size_t>(i)] =
        (gamma - 1.0) * rho[static_cast<std::size_t>(i)] *
        ei[static_cast<std::size_t>(i)];
  }

  state.rho.copy_from_host(rho);
  state.mass.copy_from_host(mass);
  state.zbar.copy_from_host(zbar);
  state.ee.copy_from_host(ee);
  state.ei.copy_from_host(ei);
  state.Te.copy_from_host(Te);
  state.Ti.copy_from_host(Ti);
  state.Pe.copy_from_host(Pe);
  state.Pi.copy_from_host(Pi);
  state.rad_E.copy_from_host(rad);
  state.volFrac.copy_from_host(vf);
  state.v_r.copy_from_host(vr);
  return state;
}

void check_same(const std::vector<double>& got,
                const std::vector<double>& expected,
                const double eps = 1.0e-12) {
  REQUIRE(got.size() == expected.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == Catch::Approx(expected[i]).epsilon(eps).margin(eps));
  }
}

}  // namespace

TEST_CASE("ALE1D driver disabled returns Disabled and leaves state unchanged",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(32);
  cfg.numerics.ale1d.enabled = false;
  auto state = make_state(cfg);
  const auto x_old = to_host(state.x_r);
  const auto mass_old = to_host(state.mass);

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);

  REQUIRE_FALSE(result.applied);
  REQUIRE(result.skip_reason ==
          tenryu::hydro::ale1d::Ale1dSkipReason::Disabled);
  check_same(to_host(state.x_r), x_old);
  check_same(to_host(state.mass), mass_old);
}

TEST_CASE("ALE1D driver does not trigger at step zero",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 100;
  auto state = make_state(cfg);
  state.step = 0;

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);

  REQUIRE_FALSE(result.applied);
  REQUIRE(result.skip_reason == tenryu::hydro::ale1d::Ale1dSkipReason::None);
}

TEST_CASE("ALE1D driver applies on cadence with benefit gate disabled",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 100;
  auto state = make_state(cfg);
  state.step = 100;

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);

  REQUIRE(result.applied);
  REQUIRE(result.skip_reason == tenryu::hydro::ale1d::Ale1dSkipReason::None);
  CHECK(state.ale_last_applied_step == 100);
}

TEST_CASE("ALE1D driver identity commit preserves conservative fields",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 100;
  auto state = make_state(cfg);
  state.step = 100;
  const auto x_old = to_host(state.x_r);
  const auto mass_old = to_host(state.mass);
  const auto ee_old = to_host(state.ee);
  const auto ei_old = to_host(state.ei);
  const auto rad_old = to_host(state.rad_E);
  const auto vf_old = to_host(state.volFrac);

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);

  REQUIRE(result.applied);
  check_same(to_host(state.x_r), x_old);
  check_same(to_host(state.mass), mass_old);
  check_same(to_host(state.ee), ee_old);
  check_same(to_host(state.ei), ei_old);
  check_same(to_host(state.rad_E), rad_old);
  check_same(to_host(state.volFrac), vf_old);
}

TEST_CASE("ALE1D driver reports mass conservation on displaced candidate",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(64);
  enable_center_sensor(cfg);
  cfg.numerics.ale1d.every_n_steps = 100;
  auto state = make_state(cfg);
  state.step = 100;

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);

  REQUIRE(result.applied);
  CHECK(result.mass_conservation_rel_err <
        cfg.numerics.ale1d.total_mass_tol.soft);
}

TEST_CASE("ALE1D driver hard diagnostic rejection discards scratch state",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(64);
  enable_center_sensor(cfg);
  cfg.numerics.ale1d.every_n_steps = 100;
  cfg.numerics.ale1d.kinetic_energy_drift_tol.hard = 0.0;
  auto state = make_state(cfg);
  state.step = 100;
  std::vector<double> vr(static_cast<std::size_t>(cfg.mesh.nr + 1), 0.0);
  for (int j = 1; j <= cfg.mesh.nr; ++j) {
    vr[static_cast<std::size_t>(j)] = 1.0e5 * static_cast<double>(j);
  }
  state.v_r.copy_from_host(vr);
  const auto x_old = to_host(state.x_r);
  const auto mass_old = to_host(state.mass);

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);

  REQUIRE_FALSE(result.applied);
  REQUIRE(result.skip_reason ==
          tenryu::hydro::ale1d::Ale1dSkipReason::ConservationRejected);
  CHECK(result.kinetic_energy_drift_rel > 0.0);
  check_same(to_host(state.x_r), x_old);
  check_same(to_host(state.mass), mass_old);
}

TEST_CASE("ALE1D driver successful commit invalidates source caches",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 100;
  auto state = make_state(cfg);
  state.step = 100;

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);

  REQUIRE(result.applied);
  CHECK(state.holo_ale_invalidated);
  CHECK(state.ale_rezoned);
}

TEST_CASE("ALE1D driver applies fixed boundary after commit",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(64);
  enable_center_sensor(cfg);
  cfg.numerics.hydro.boundary_1d = "fixed";
  cfg.numerics.ale1d.every_n_steps = 100;
  auto state = make_state(cfg);
  state.step = 100;

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  const auto x = to_host(state.x_r);

  REQUIRE(result.applied);
  CHECK(x.front() == Catch::Approx(cfg.mesh.r_min).margin(1.0e-14));
  CHECK(x.back() == Catch::Approx(cfg.mesh.r_max).margin(1.0e-14));
}

TEST_CASE("1D hydro driver runs with ALE1D enabled but untriggered",
          "[hydro][ale1d][driver][coupling]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(24);
  cfg.main.t_end = 3.0e-12;
  cfg.main.max_steps = 3;
  cfg.numerics.dt.initial_s = 1.0e-12;
  cfg.numerics.dt.max_s = 1.0e-12;
  cfg.numerics.dt.min_s = 1.0e-20;
  cfg.numerics.ale1d.every_n_steps = 1000000;
  cfg.output.directory = "./test_output_ale1d_driver_skip";
  cfg.output.plot_every = 0;
  cfg.output.history_every = 0;
  cfg.output.checkpoint_every = 0;
  cfg.output.plot_every_s = -1.0;
  cfg.output.history_every_s = -1.0;
  cfg.output.checkpoint_every_s = -1.0;

  auto state = make_state(cfg);
  tenryu::coupling::Driver driver;
  driver.run(state, cfg);

  CHECK(state.step > 0);
  CHECK_FALSE(state.ale_rezoned);
  CHECK(state.ale_last_applied_step == -1);
}

namespace {

constexpr double kTableTTopEv = 2000.0;  // build_power_law_te_tables T ceiling
constexpr double kTableTMinEv = 0.05;    // build_power_law_te_tables T floor row

tenryu::core::Config make_table_eos_cfg(const int n) {
  auto cfg = make_cfg(n);
  auto& mat = cfg.materials.materials.front();
  mat.eos_model = "power_law_te";
  // e_e = f * T (beta=1, mu_rho=0): linear, monotone, rho-independent.
  mat.eos_tables = std::make_shared<const tenryu::materials::EOSTableTriplet>(
      tenryu::materials::build_power_law_te_tables(
          mat.A, mat.ideal_gas_gamma, 1.0e12, 1.0, 0.0, 5.0 / 3.0));
  return cfg;
}

}  // namespace

TEST_CASE("ALE1D reclosure keeps super-ceiling energy under energy_authoritative",
          "[hydro][ale1d][driver][eos_closure]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_table_eos_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 100;
  REQUIRE(cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
  tenryu::hydro::HydroEOSContext ctx;
  ctx.initialize(cfg);

  auto state = make_state(cfg);
  state.step = 100;
  const double cv_i = cv_mass(cfg);
  const int hot = 7;
  auto ei_host = to_host(state.ei);
  auto ee_host = to_host(state.ee);
  const double e_super = cv_i * 1.5 * kTableTTopEv;  // above the table ceiling
  ei_host[hot] = e_super;
  ee_host[hot] = 1.0e12;  // T_e = 1 eV, in-table
  state.ei.copy_from_host(ei_host);
  state.ee.copy_from_host(ee_host);
  const auto ei_before = to_host(state.ei);

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg, &ctx);
  REQUIRE(result.applied);

  const auto ei_after = to_host(state.ei);
  const auto Ti_after = to_host(state.Ti);
  // Super-ceiling cell: energy preserved, temperature extends into the ideal
  // tail (e linear in T for the ion table => T_tail = 1.5 * T_top).
  CHECK(ei_after[hot] == ei_before[hot]);
  CHECK(Ti_after[hot] ==
        Catch::Approx(1.5 * kTableTTopEv).epsilon(1.0e-10));
  // Below-table cells (e = cv*1e-3 < e(T_min)): evolved energy kept, T at the
  // table lower edge.
  const int cold = 3;
  CHECK(ei_after[cold] == ei_before[cold]);
  CHECK(to_host(state.Ti)[cold] ==
        Catch::Approx(kTableTMinEv).epsilon(1.0e-10));
  CHECK(state.E_floor_injected == 0.0);
}

TEST_CASE("ALE1D reclosure legacy mode keeps the historic table projection",
          "[hydro][ale1d][driver][eos_closure]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_table_eos_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 100;
  cfg.numerics.hydro.eos_closure_mode = "legacy";
  tenryu::hydro::HydroEOSContext ctx;
  ctx.initialize(cfg);

  auto state = make_state(cfg);
  state.step = 100;
  const double cv_i = cv_mass(cfg);
  const int hot = 7;
  auto ei_host = to_host(state.ei);
  auto ee_host = to_host(state.ee);
  ei_host[hot] = cv_i * 1.5 * kTableTTopEv;
  ee_host[hot] = 1.0e12;
  state.ei.copy_from_host(ei_host);
  state.ee.copy_from_host(ee_host);

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg, &ctx);
  REQUIRE(result.applied);

  const auto ei_after = to_host(state.ei);
  // Legacy: super-ceiling energy projected down to the table ceiling and the
  // sub-table cells raised to the T_min row value (historic behavior).
  CHECK(ei_after[hot] ==
        Catch::Approx(cv_i * kTableTTopEv).epsilon(1.0e-10));
  const int cold = 3;
  CHECK(ei_after[cold] ==
        Catch::Approx(cv_i * kTableTMinEv).epsilon(1.0e-3));
}

// The commit closes the EOS with the hydro's closure. Under the default
// energy_authoritative closure a table cell below the temperature floor keeps
// its evolved energy and reports T at the floor (the clamp veto of the
// hydro's 2T table closure); nothing is injected. The ALE's own reclosure
// raised such energies to e(T_floor) and booked the difference.
TEST_CASE("ALE1D commit keeps sub-floor table energies like the hydro closure",
          "[hydro][ale1d][driver][eos_closure]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_table_eos_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 100;
  cfg.numerics.floors.Ti = 0.1;  // inside the table domain (> T_min row)
  REQUIRE(cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
  tenryu::hydro::HydroEOSContext ctx;
  ctx.initialize(cfg);

  auto state = make_state(cfg);
  state.step = 100;
  const double cv_i = cv_mass(cfg);
  auto ei_host = to_host(state.ei);
  auto ee_host = to_host(state.ee);
  // All cells: T_i(e) ~ 0.06 eV, below the 0.1 eV floor but inside the table;
  // T_e = 1 eV, above the floor.
  for (std::size_t c = 0; c < ei_host.size(); ++c) {
    ei_host[c] = cv_i * 0.06;
    ee_host[c] = 1.0e12;
  }
  state.ei.copy_from_host(ei_host);
  state.ee.copy_from_host(ee_host);
  const auto ei_before = to_host(state.ei);

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg, &ctx);
  REQUIRE(result.applied);

  const auto ei_after = to_host(state.ei);
  const auto Ti_after = to_host(state.Ti);
  for (std::size_t c = 0; c < ei_after.size(); ++c) {
    INFO("cell " << c);
    CHECK(ei_after[c] == Catch::Approx(ei_before[c]).epsilon(1.0e-12));
    CHECK(Ti_after[c] == Catch::Approx(0.1).epsilon(1.0e-12));
  }
  CHECK(state.E_floor_injected == 0.0);
}

// 1T: e_e holds the total specific energy, e_i = 0, T_i = T_e =
// e / (c_v,i + c_v,e), P_e the total pressure, c_v,e the total heat capacity.
// The ALE's own reclosure closed the ion and electron parts separately (the
// 2T formulas): T_i at the floor with c_v,i T_floor put into e_i (and booked
// as a floor injection), T_e = e / c_v,e, and c_v,e without the ion part.
TEST_CASE("ALE1D commit closes a 1T state with the 1T closure",
          "[hydro][ale1d][driver][eos_closure]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(32);
  cfg.main.two_temperature = false;
  cfg.numerics.ale1d.every_n_steps = 100;

  auto state = make_state(cfg);
  state.step = 100;
  const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
  const double cv_total = 2.0 * cv_mass(cfg);  // (1 + zbar) with zbar = 1
  constexpr double kT = 5.0;                   // eV, above the floors
  const std::size_t n = static_cast<std::size_t>(cfg.mesh.nr);
  state.ee.copy_from_host(std::vector<double>(n, cv_total * kT));
  state.ei.copy_from_host(std::vector<double>(n, 0.0));
  state.Te.copy_from_host(std::vector<double>(n, kT));
  state.Ti.copy_from_host(std::vector<double>(n, kT));
  const double floor_before = state.E_floor_injected;

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  REQUIRE(result.applied);

  const auto ee = to_host(state.ee);
  const auto ei = to_host(state.ei);
  const auto Te = to_host(state.Te);
  const auto Ti = to_host(state.Ti);
  const auto Pe = to_host(state.Pe);
  const auto Pi = to_host(state.Pi);
  const auto rho = to_host(state.rho);
  const auto cs = to_host(state.cs);
  for (std::size_t c = 0; c < n; ++c) {
    INFO("cell " << c << " ee " << ee[c] << " ei " << ei[c] << " Te " << Te[c] << " Ti "
                 << Ti[c]);
    CHECK(ei[c] == 0.0);
    CHECK(Te[c] == Catch::Approx(ee[c] / cv_total).epsilon(1.0e-12));
    CHECK(Ti[c] == Te[c]);
    CHECK(Pe[c] == Catch::Approx((gamma - 1.0) * rho[c] * ee[c]).epsilon(1.0e-12));
    CHECK(Pi[c] == 0.0);
    CHECK(cs[c] == Catch::Approx(std::sqrt(gamma * (gamma - 1.0) * ee[c])).epsilon(1.0e-12));
  }
  CHECK(state.E_floor_injected == floor_before);
  // The 1T closure keeps the total heat capacity in c_v,e.
  REQUIRE(state.cv_e.size() == n);
  const auto cv_e = to_host(state.cv_e);
  for (std::size_t c = 0; c < n; ++c) {
    CHECK(cv_e[c] == Catch::Approx(cv_total).epsilon(1.0e-12));
  }
}

TEST_CASE("ALE1D acoustic time-step bounds of the current and candidate meshes",
          "[hydro][ale1d][driver][gates]") {
  using tenryu::hydro::ale1d::acoustic_dt_bounds;
  const std::vector<double> r{0.0, 1.0, 2.0, 3.0};
  const std::vector<double> cs{1.0, 2.0, 1.0};
  // Current: min(1/1, 1/2, 1/1). Candidate [0,0.5],[0.5,2.5],[2.5,3]: the
  // middle cell overlaps all three current cells (largest c = 2).
  auto bounds = acoustic_dt_bounds(r, cs, {0.0, 0.5, 2.5, 3.0});
  CHECK(bounds.current == 0.5);
  CHECK(bounds.candidate == 0.5);
  // [0,1.5] overlaps cells 0 and 1 (0.75), [1.5,2] cell 1 (0.25), [2,3] cell 2.
  bounds = acoustic_dt_bounds(r, cs, {0.0, 1.5, 2.0, 3.0});
  CHECK(bounds.candidate == 0.25);
  // A candidate face on a current node does not reach into the next cell.
  bounds = acoustic_dt_bounds(r, cs, {0.0, 1.0, 2.5, 3.0});
  CHECK(bounds.candidate == 0.5);
  // No sound speed: no bound.
  bounds = acoustic_dt_bounds(r, {0.0, 0.0, 0.0}, {0.0, 0.5, 2.5, 3.0});
  CHECK(std::isinf(bounds.current));
  CHECK(std::isinf(bounds.candidate));
}

namespace {

// A graded mesh: cell widths grow by `ratio` outward (largest adjacent ratio =
// ratio), with the geometry, masses and pressures made consistent.
void grade_mesh(tenryu::core::State& state, const tenryu::core::Config& cfg,
                const double ratio) {
  const int n = cfg.mesh.nr;
  std::vector<double> r(static_cast<std::size_t>(n + 1), 0.0);
  double width = (ratio - 1.0) / (std::pow(ratio, n) - 1.0);
  for (int j = 1; j <= n; ++j) {
    r[static_cast<std::size_t>(j)] = r[static_cast<std::size_t>(j - 1)] + width;
    width *= ratio;
  }
  r.back() = 1.0;
  state.x_r.copy_from_host(r);
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;
  const auto vol = to_host(state.vol);
  const auto rho = to_host(state.rho);
  std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    mass[static_cast<std::size_t>(i)] =
        rho[static_cast<std::size_t>(i)] * vol[static_cast<std::size_t>(i)];
  }
  state.mass.copy_from_host(mass);
}

void set_sound_speed(tenryu::core::State& state, const double cs) {
  state.cs.reset(state.rho.size());
  state.cs.copy_from_host(std::vector<double>(state.rho.size(), cs));
}

}  // namespace

TEST_CASE("ALE1D min_steps_between_ale skip is reported as TooSoon",
          "[hydro][ale1d][driver][gates]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 100;
  cfg.numerics.ale1d.min_steps_between_ale = 50;
  auto state = make_state(cfg);
  state.step = 100;
  state.ale_last_applied_step = 80;
  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  REQUIRE_FALSE(result.applied);
  CHECK(result.cadence_triggered);
  CHECK(result.skip_reason == tenryu::hydro::ale1d::Ale1dSkipReason::TooSoon);
}

// The mesh-quality trigger compares the largest adjacent width ratio with
// emergency_max_dr_ratio (it used candidate_dt_penalty_max).
TEST_CASE("ALE1D quality trigger uses emergency_max_dr_ratio",
          "[hydro][ale1d][driver][gates]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 1000000;
  cfg.numerics.ale1d.candidate_dt_penalty_max = 10.0;
  auto state = make_state(cfg);
  grade_mesh(state, cfg, 1.1);
  state.step = 7;

  cfg.numerics.ale1d.emergency_max_dr_ratio = 1.05;
  auto low = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  INFO("largest adjacent ratio " << low.max_dr_ratio);
  CHECK(low.max_dr_ratio == Catch::Approx(1.1).epsilon(1.0e-6));
  CHECK(low.quality_triggered);

  auto state2 = make_state(cfg);
  grade_mesh(state2, cfg, 1.1);
  state2.step = 7;
  cfg.numerics.ale1d.emergency_max_dr_ratio = 1.2;
  cfg.numerics.ale1d.candidate_dt_penalty_max = 1.0;  // no longer the trigger threshold
  const auto high = tenryu::hydro::ale1d::apply_ale_1d(state2, cfg);
  CHECK_FALSE(high.quality_triggered);
  CHECK_FALSE(high.applied);
  CHECK(high.skip_reason == tenryu::hydro::ale1d::Ale1dSkipReason::None);
}

// A candidate that refines the centre shortens the acoustic step: it is
// rejected when that exceeds candidate_dt_penalty_max, and a cadence-only
// attempt with the benefit gate on is rejected for no gain; a mesh-quality
// attempt is exempt from the benefit gate.
TEST_CASE("ALE1D candidate gates on the acoustic time step",
          "[hydro][ale1d][driver][gates]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(64);
  enable_center_sensor(cfg);
  cfg.numerics.ale1d.every_n_steps = 100;

  auto penalty_cfg = cfg;
  penalty_cfg.numerics.ale1d.candidate_dt_penalty_max = 1.0;
  auto state = make_state(penalty_cfg);
  set_sound_speed(state, 1.0e5);
  state.step = 100;
  const auto x_before = to_host(state.x_r);
  const auto penalized = tenryu::hydro::ale1d::apply_ale_1d(state, penalty_cfg);
  INFO("candidate dt gain " << penalized.candidate_dt_gain);
  REQUIRE_FALSE(penalized.applied);
  CHECK(penalized.skip_reason ==
        tenryu::hydro::ale1d::Ale1dSkipReason::DtPenaltyTooLarge);
  CHECK(penalized.candidate_dt_gain < 1.0);
  check_same(to_host(state.x_r), x_before);

  auto allowed_cfg = cfg;
  allowed_cfg.numerics.ale1d.candidate_dt_penalty_max = 100.0;
  auto state2 = make_state(allowed_cfg);
  set_sound_speed(state2, 1.0e5);
  state2.step = 100;
  const auto allowed = tenryu::hydro::ale1d::apply_ale_1d(state2, allowed_cfg);
  CHECK(allowed.applied);
  CHECK(allowed.candidate_dt_gain == Catch::Approx(penalized.candidate_dt_gain));

  auto benefit_cfg = allowed_cfg;
  benefit_cfg.numerics.ale1d.enable_benefit_gate = true;
  auto state3 = make_state(benefit_cfg);
  set_sound_speed(state3, 1.0e5);
  state3.step = 100;
  const auto no_gain = tenryu::hydro::ale1d::apply_ale_1d(state3, benefit_cfg);
  REQUIRE_FALSE(no_gain.applied);
  CHECK(no_gain.skip_reason == tenryu::hydro::ale1d::Ale1dSkipReason::BenefitTooSmall);

  // The same candidate on a graded mesh that also fires the quality trigger:
  // a rescue attempt, exempt from the benefit gate.
  auto rescue_cfg = benefit_cfg;
  rescue_cfg.numerics.ale1d.emergency_max_dr_ratio = 1.01;
  auto state4 = make_state(rescue_cfg);
  grade_mesh(state4, rescue_cfg, 1.02);
  set_sound_speed(state4, 1.0e5);
  state4.step = 100;
  const auto rescue = tenryu::hydro::ale1d::apply_ale_1d(state4, rescue_cfg);
  INFO("rescue skip reason " << tenryu::hydro::ale1d::to_string(rescue.skip_reason)
                             << " gain " << rescue.candidate_dt_gain);
  CHECK(rescue.quality_triggered);
  CHECK(rescue.skip_reason != tenryu::hydro::ale1d::Ale1dSkipReason::BenefitTooSmall);
}

TEST_CASE("ALE1D min-width-floor cooldown suppresses evaluation after rejected attempt",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  auto cfg = make_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 1000000;
  cfg.numerics.ale1d.emergency_enabled = false;
  cfg.numerics.ale1d.min_width_floor.enabled = true;
  // Uniform dx = 1/32 cm: every cell is under the floor and the span-preserving
  // window cannot improve a uniform mesh, so the attempt is never applied.
  cfg.numerics.ale1d.min_width_floor.floor_cm = 0.05;
  cfg.numerics.ale1d.min_width_floor.retrigger_cooldown_steps = 2;
  auto state = make_state(cfg);

  state.step = 1;
  const auto first = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  REQUIRE(first.floor_triggered);
  REQUIRE_FALSE(first.applied);
  REQUIRE(state.ale1d_floor_cooldown_remaining == 2);

  state.step = 2;
  const auto second = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  REQUIRE_FALSE(second.floor_triggered);
  REQUIRE_FALSE(second.applied);
  REQUIRE(state.ale1d_floor_cooldown_remaining == 1);

  state.step = 3;
  const auto third = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  REQUIRE_FALSE(third.floor_triggered);
  REQUIRE(state.ale1d_floor_cooldown_remaining == 0);

  state.step = 4;
  const auto fourth = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  REQUIRE(fourth.floor_triggered);
  REQUIRE_FALSE(fourth.applied);
  REQUIRE(state.ale1d_floor_cooldown_remaining == 2);
}

TEST_CASE("ALE1D min-width-floor cooldown default keeps per-step evaluation and applied rezone resets it",
          "[hydro][ale1d][driver]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  // Default retrigger_cooldown_steps=0: the floor trigger is evaluated every step.
  auto cfg = make_cfg(32);
  cfg.numerics.ale1d.every_n_steps = 1000000;
  cfg.numerics.ale1d.emergency_enabled = false;
  cfg.numerics.ale1d.min_width_floor.enabled = true;
  cfg.numerics.ale1d.min_width_floor.floor_cm = 0.05;
  auto state = make_state(cfg);

  state.step = 1;
  const auto first = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  REQUIRE(first.floor_triggered);
  REQUIRE_FALSE(first.applied);
  REQUIRE(state.ale1d_floor_cooldown_remaining == 0);

  state.step = 2;
  const auto second = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);
  REQUIRE(second.floor_triggered);
  REQUIRE(state.ale1d_floor_cooldown_remaining == 0);

  // An applied rezone resets an active cooldown.
  auto cfg2 = make_cfg(32);
  cfg2.numerics.ale1d.every_n_steps = 100;
  cfg2.numerics.ale1d.min_width_floor.enabled = true;
  cfg2.numerics.ale1d.min_width_floor.floor_cm = 1.0e-9;  // never triggers
  cfg2.numerics.ale1d.min_width_floor.retrigger_cooldown_steps = 7;
  auto state2 = make_state(cfg2);
  state2.step = 100;
  state2.ale1d_floor_cooldown_remaining = 5;  // as if a floor attempt was rejected earlier
  const auto applied = tenryu::hydro::ale1d::apply_ale_1d(state2, cfg2);
  REQUIRE(applied.applied);
  REQUIRE(state2.ale1d_floor_cooldown_remaining == 0);
}
