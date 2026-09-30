#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/state.hpp"
#include "radiation/fleck.cuh"

namespace {

// One ideal-gas cell with a constant absorption opacity. Without a table cv_e
// and with cv_e_override <= 0, compute_dt_rad_limit uses the ideal-gas
// fallback Cv_e = rho * zbar * eV_to_erg / (A m_p (gamma - 1)) and caps beta
// at 1.
struct Setup {
  tenryu::core::Config cfg;
  tenryu::core::State state;
};

Setup make_setup(const tenryu::core::RadiationMode mode) {
  Setup s;
  tenryu::core::Config::MaterialsConfig::MatDef mat;
  mat.name = "m";
  mat.A = 1.0;
  mat.ideal_gas_gamma = 5.0 / 3.0;
  mat.opacity_model = "constant";
  mat.kappa_a_constant = 1.0;  // [cm^2/g]
  s.cfg.materials.materials = {mat};
  s.cfg.radiation.enabled = true;
  s.cfg.radiation.mode = mode;
  s.cfg.radiation.groups = 1;
  s.cfg.radiation.imc.alpha = 1.0;
  s.cfg.numerics.dt.f_min_fleck = 0.01;
  s.state.rho = std::vector<double>{1.0};
  s.state.Te = std::vector<double>{10.0};
  s.state.zbar = std::vector<double>{1.0};
  s.state.cell_is_void.assign(1, 0U);
  return s;
}

}  // namespace

// Numerics.dt.f_min_fleck is an IMC-only time-step constraint (NUMERICS
// §2.2 (c), SPECIFICATION §6.4.7): the FLD and S_N modes must get +inf and
// only imc_ddmc the Fleck-floor value.
TEST_CASE("compute_dt_rad_limit applies the Fleck-factor floor to IMC only",
          "[radiation][fleck][dt]") {
  using tenryu::core::RadiationMode;
  namespace k = tenryu::core::constants;

  SECTION("multigroup diffusion (FLD): no constraint") {
    Setup s = make_setup(RadiationMode::MultigroupDiffusion);
    tenryu::radiation::DtRadDiagnostics diag;
    const double dt =
        tenryu::radiation::compute_dt_rad_limit(s.state, s.cfg, &diag);
    CHECK(std::isinf(dt));
    CHECK(diag.limiting_cell == -1);
  }

  SECTION("S_N transport: no constraint") {
    Setup s = make_setup(RadiationMode::SnTransport);
    const double dt = tenryu::radiation::compute_dt_rad_limit(s.state, s.cfg);
    CHECK(std::isinf(dt));
  }

  SECTION("IMC: dt_rad = (1 - f_min) / (f_min alpha c beta sigma_P)") {
    Setup s = make_setup(RadiationMode::ImcDdmc);
    tenryu::radiation::DtRadDiagnostics diag;
    const double dt =
        tenryu::radiation::compute_dt_rad_limit(s.state, s.cfg, &diag);
    const double rho = 1.0;
    const double Te = 10.0;
    const double zbar = 1.0;
    const double A = 1.0;
    const double gm1 = 5.0 / 3.0 - 1.0;
    const double Cv_e = rho * zbar * k::eV_to_erg / (A * k::proton_mass * gm1);
    const double beta = std::min(4.0 * k::a_eV * Te * Te * Te / Cv_e, 1.0);
    const double sigma_P = rho * 1.0;
    const double expected =
        (1.0 - 0.01) / (0.01 * 1.0 * k::c_light * beta * sigma_P);
    REQUIRE(std::isfinite(dt));
    CHECK(dt == Catch::Approx(expected).epsilon(1.0e-9));
    CHECK(diag.limiting_cell == 0);
  }
}
