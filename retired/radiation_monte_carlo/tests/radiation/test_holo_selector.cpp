#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/holo_selector.hpp"
#include "radiation/planck_table.cuh"

namespace {

struct SelectorFixture {
  tenryu::radiation::HoloSelectorConfig cfg;
  tenryu::radiation::HoloSelectorInputs in;
  std::vector<double> node_r;
  std::vector<double> mass;
  std::vector<double> Te;
  std::vector<double> sigma_R;
  tenryu::radiation::PlanckTable planck;
  std::vector<std::uint8_t> cell_is_void;
  std::vector<std::uint8_t> core_mask;
  std::vector<std::uint8_t> patch_mask;
  std::vector<std::uint8_t> prev_core_mask;
  std::vector<std::int32_t> hold_count;
  std::vector<std::int32_t> dwell_count;
  std::vector<double> tau_R;
  std::vector<double> reduced_flux;
  std::vector<double> mass_q;
  std::vector<double> lo_weight;
  bool valid = false;

  SelectorFixture(const int n_cells, const int n_groups)
      : node_r(static_cast<std::size_t>(n_cells + 1), 0.0),
        mass(static_cast<std::size_t>(n_cells), 1.0),
        Te(static_cast<std::size_t>(n_cells), 10.0),
        sigma_R(static_cast<std::size_t>(n_cells * n_groups), 10.0),
        cell_is_void(static_cast<std::size_t>(n_cells), 0U) {
    cfg.enabled = true;
    cfg.coupling_tau = 5.0;
    cfg.guard_cells = 0;
    cfg.blend_cells = 0;
    cfg.min_lo_cells = 1;
    for (int i = 0; i <= n_cells; ++i) {
      node_r[static_cast<std::size_t>(i)] = static_cast<double>(i);
    }
    in.n_cells = n_cells;
    in.n_groups = n_groups;
    in.step = 0;
    in.node_r = &node_r;
    in.mass = &mass;
    in.Te = &Te;
    in.sigma_R = &sigma_R;
    in.planck = &planck;
    in.cell_is_void = &cell_is_void;
  }

  tenryu::radiation::HoloSelectorDiagnostics update() {
    tenryu::radiation::HoloSelectorStateView state{core_mask,
                                                   patch_mask,
                                                   prev_core_mask,
                                                   hold_count,
                                                   dwell_count,
                                                   tau_R,
                                                   reduced_flux,
                                                   mass_q,
                                                   lo_weight,
                                                   valid};
    return tenryu::radiation::update_holo_core_mask(cfg, in, state);
  }
};

}  // namespace

TEST_CASE("HOLO selector marks cells above local optical depth threshold",
          "[radiation][holo]") {
  SelectorFixture f(4, 1);
  f.sigma_R = {1.0, 6.0, 4.0, 7.0};

  const auto diag = f.update();

  REQUIRE(diag.n_core_cells == 2);
  REQUIRE(diag.n_patch_cells == 2);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({0, 1, 0, 1}));
  REQUIRE(f.patch_mask == std::vector<std::uint8_t>({0, 1, 0, 1}));
  REQUIRE(f.lo_weight == std::vector<double>({0.0, 1.0, 0.0, 1.0}));
  REQUIRE(f.tau_R == std::vector<double>({1.0, 6.0, 4.0, 7.0}));
}

TEST_CASE("HOLO selector expands patch mask by guard cells",
          "[radiation][holo]") {
  SelectorFixture f(6, 1);
  f.cfg.guard_cells = 1;
  f.sigma_R = {1.0, 1.0, 6.0, 1.0, 1.0, 1.0};

  const auto diag = f.update();

  REQUIRE(diag.n_core_cells == 1);
  REQUIRE(diag.n_patch_cells == 3);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({0, 0, 1, 0, 0, 0}));
  REQUIRE(f.patch_mask == std::vector<std::uint8_t>({0, 1, 1, 1, 0, 0}));
  REQUIRE(f.lo_weight == std::vector<double>({0.0, 0.0, 1.0, 0.0, 0.0, 0.0}));
}

TEST_CASE("HOLO selector excludes void cells from coupling mask",
          "[radiation][holo]") {
  SelectorFixture f(3, 1);
  f.cfg.guard_cells = 1;
  f.sigma_R = {6.0, 6.0, 6.0};
  f.cell_is_void[1] = 1U;

  const auto diag = f.update();

  REQUIRE(diag.n_core_cells == 2);
  REQUIRE(diag.n_patch_cells == 2);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({1, 0, 1}));
  REQUIRE(f.patch_mask == std::vector<std::uint8_t>({1, 0, 1}));
}

TEST_CASE("HOLO selector reports entry and exit against previous mask",
          "[radiation][holo]") {
  SelectorFixture f(3, 1);
  auto diag = f.update();
  REQUIRE(diag.n_core_cells == 3);

  f.sigma_R = {1.0, 6.0, 1.0};
  f.in.step = 1;
  diag = f.update();

  REQUIRE(diag.n_entered == 0);
  REQUIRE(diag.n_exited == 2);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({0, 1, 0}));
}

TEST_CASE("HOLO selector uses tau hysteresis for core entry and exit",
          "[radiation][holo]") {
  SelectorFixture f(3, 1);
  f.cfg.tau_on = 5.0;
  f.cfg.tau_off = 3.0;
  f.sigma_R = {6.0, 4.0, 2.0};

  auto diag = f.update();

  REQUIRE(diag.n_core_cells == 1);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({1, 0, 0}));
  REQUIRE(f.dwell_count == std::vector<std::int32_t>({1, 0, 0}));

  f.sigma_R = {4.0, 4.0, 2.0};
  f.in.step = 1;
  diag = f.update();

  REQUIRE(diag.n_entered == 0);
  REQUIRE(diag.n_exited == 0);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({1, 0, 0}));
  REQUIRE(f.dwell_count == std::vector<std::int32_t>({2, 0, 0}));

  f.sigma_R = {2.0, 4.0, 2.0};
  f.in.step = 2;
  diag = f.update();

  REQUIRE(diag.n_entered == 0);
  REQUIRE(diag.n_exited == 1);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({0, 0, 0}));
  REQUIRE(f.dwell_count == std::vector<std::int32_t>({0, 0, 0}));
}

TEST_CASE("HOLO selector enforces minimum core dwell time",
          "[radiation][holo]") {
  SelectorFixture f(1, 1);
  f.cfg.tau_on = 5.0;
  f.cfg.tau_off = 3.0;
  f.cfg.min_dwell_steps = 2;
  f.sigma_R = {6.0};

  auto diag = f.update();

  REQUIRE(diag.n_core_cells == 1);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({1}));
  REQUIRE(f.dwell_count == std::vector<std::int32_t>({1}));

  f.sigma_R = {1.0};
  f.in.step = 1;
  diag = f.update();

  REQUIRE(diag.n_exited == 0);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({1}));
  REQUIRE(f.dwell_count == std::vector<std::int32_t>({2}));

  f.in.step = 2;
  diag = f.update();

  REQUIRE(diag.n_exited == 1);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({0}));
  REQUIRE(f.dwell_count == std::vector<std::int32_t>({0}));
}

TEST_CASE("HOLO selector assigns blend weights by core distance",
          "[radiation][holo]") {
  SelectorFixture f(7, 1);
  f.cfg.blend_cells = 3;
  f.sigma_R = {1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 1.0};

  const auto diag = f.update();

  REQUIRE(diag.n_core_cells == 1);
  REQUIRE(diag.n_patch_cells == 7);
  REQUIRE(diag.n_blend_cells == 4);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({0, 0, 0, 1, 0, 0, 0}));
  REQUIRE(f.patch_mask == std::vector<std::uint8_t>({1, 1, 1, 1, 1, 1, 1}));
  REQUIRE(f.lo_weight[0] == Catch::Approx(0.0));
  REQUIRE(f.lo_weight[1] == Catch::Approx(1.0 / 3.0));
  REQUIRE(f.lo_weight[2] == Catch::Approx(2.0 / 3.0));
  REQUIRE(f.lo_weight[3] == Catch::Approx(1.0));
  REQUIRE(f.lo_weight[4] == Catch::Approx(2.0 / 3.0));
  REQUIRE(f.lo_weight[5] == Catch::Approx(1.0 / 3.0));
  REQUIRE(f.lo_weight[6] == Catch::Approx(0.0));
}

TEST_CASE("HOLO selector disables small LO cores",
          "[radiation][holo]") {
  SelectorFixture f(4, 1);
  f.cfg.min_lo_cells = 2;
  f.cfg.guard_cells = 1;
  f.cfg.blend_cells = 2;
  f.sigma_R = {1.0, 6.0, 1.0, 1.0};

  const auto diag = f.update();

  REQUIRE(diag.active);
  REQUIRE(diag.n_core_cells == 0);
  REQUIRE(diag.n_patch_cells == 0);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({0, 0, 0, 0}));
  REQUIRE(f.patch_mask == std::vector<std::uint8_t>({0, 0, 0, 0}));
  REQUIRE(f.lo_weight == std::vector<double>({0.0, 0.0, 0.0, 0.0}));
}

TEST_CASE("HOLO selector keeps unsupported geometry inactive in v1",
          "[radiation][holo]") {
  SelectorFixture f(3, 1);
  f.core_mask = {1U, 1U, 1U};
  f.valid = true;
  f.in.dimension = tenryu::radiation::HoloGeometryDimension::RZ2D;

  const auto diag = f.update();

  REQUIRE_FALSE(diag.active);
  REQUIRE_FALSE(f.valid);
  REQUIRE(f.core_mask == std::vector<std::uint8_t>({0, 0, 0}));
  REQUIRE(f.patch_mask == std::vector<std::uint8_t>({0, 0, 0}));
  REQUIRE(f.lo_weight == std::vector<double>({0.0, 0.0, 0.0}));
}
