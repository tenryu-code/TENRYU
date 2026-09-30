// Retired with the Monte Carlo radiation on 2026-09-29: the test case of tests/hydro/test_ale_1d_skeleton.cu (as it
// was at 5bc8f6ce3) that checked the 1D ALE configuration check rejects Radiation.mode "imc_ddmc". Not built.

TEST_CASE("ConfigError when ale1d.enabled with IMC mode",
          "[hydro][ale1d]") {
  tenryu::core::Config cfg;
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.numerics.ale1d.enabled = true;
  cfg.radiation.mode = tenryu::core::RadiationMode::ImcDdmc;

  REQUIRE_THROWS_AS(tenryu::core::validate_ale1d_config(cfg),
                    tenryu::core::namelist::ConfigError);
}
