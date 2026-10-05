#include <array>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/config_validate.hpp"
#include "core/namelist/errors.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_driver.cuh"

namespace {

tenryu::core::State make_state(const int dim, const int n_cells) {
  tenryu::core::State state;
  state.mesh.dim = dim;
  state.mesh.topo.nr = n_cells;
  state.mesh.topo.nz = (dim == 2) ? 2 : 1;
  state.mesh.topo.n_cells = n_cells;
  state.mesh.topo.n_nodes = (dim == 2) ? ((n_cells + 1) * 3) : (n_cells + 1);
  state.step = 0;
  return state;
}

}  // namespace

TEST_CASE("Ale1dSkipReason::Disabled when ale1d.enabled=false",
          "[hydro][ale1d]") {
  tenryu::core::Config cfg;
  auto state = make_state(1, 256);

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);

  REQUIRE(result.applied == false);
  REQUIRE(result.skip_reason == tenryu::hydro::ale1d::Ale1dSkipReason::Disabled);
}

TEST_CASE("Ale1dSkipReason::WrongGeometry when 2D mesh",
          "[hydro][ale1d]") {
  tenryu::core::Config cfg;
  cfg.main.dimension = "2D_RZ";
  cfg.main.dim = 2;
  cfg.numerics.ale1d.enabled = true;
  auto state = make_state(2, 256);

  const auto result = tenryu::hydro::ale1d::apply_ale_1d(state, cfg);

  REQUIRE(result.applied == false);
  REQUIRE(result.skip_reason == tenryu::hydro::ale1d::Ale1dSkipReason::WrongGeometry);
}

TEST_CASE("Ale1dStepResult default values", "[hydro][ale1d]") {
  tenryu::hydro::ale1d::Ale1dStepResult r;

  REQUIRE(r.applied == false);
  REQUIRE(r.skip_reason == tenryu::hydro::ale1d::Ale1dSkipReason::None);
  REQUIRE(r.kinetic_energy_drift_rel == Catch::Approx(0.0));
}

TEST_CASE("to_string(Ale1dSkipReason) returns non-empty for all values",
          "[hydro][ale1d]") {
  using tenryu::hydro::ale1d::Ale1dSkipReason;
  constexpr std::array reasons{
      Ale1dSkipReason::None,
      Ale1dSkipReason::Disabled,
      Ale1dSkipReason::WrongGeometry,
      Ale1dSkipReason::NTooSmall,
      Ale1dSkipReason::ProtectedFractionTooHigh,
      Ale1dSkipReason::MovableSegmentTooSmall,
      Ale1dSkipReason::BenefitTooSmall,
      Ale1dSkipReason::CandidateInvalid,
      Ale1dSkipReason::ConservationRejected,
      Ale1dSkipReason::DtPenaltyTooLarge,
  };

  for (const auto reason : reasons) {
    REQUIRE_FALSE(std::string(tenryu::hydro::ale1d::to_string(reason)).empty());
  }
}
