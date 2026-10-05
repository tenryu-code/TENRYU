// Extract (a reference copy, not a compilable unit): the 1D ALE test cases of test files that stayed in the
// build, as they were at 6d62bf929. They ran in the executables test_namelist_ai_review_1d_keys and
// test_namelist_radiation_guards (tests/core/CMakeLists.txt); the file-local helpers they used
// (make_1d_config, ensure_python_started, make_validation_config, validate_with_builder,
// require_config_error_contains) are in those files.

// ---- tests/core/test_namelist_ai_review_1d_keys.cpp, lines 217-267 at 6d62bf929 ----
TEST_CASE("ALE1D V3 scope guards fail closed",
          "[core][namelist][ai_review_1d][ale1d]") {
  // The guards are unit-tested through validate_ale1d_config directly: the
  // full builder validation trips unrelated per-EOS checks first (e.g.
  // eos.file is required for eos_model="tmat") before reaching the ALE1D
  // scope guards, which is not what this case is about. The builder-level
  // path is exercised by the 2D_RZ rejection test above.
  auto make_ale = [] {
    auto cfg = make_1d_config();
    cfg.numerics.ale1d.enabled = true;
    return cfg;
  };

  {
    auto cfg = make_ale();
    REQUIRE_NOTHROW(tenryu::core::validate_ale1d_config(cfg));
  }
  {
    auto cfg = make_ale();
    tenryu::core::Config::MaterialsConfig::MatDef second;
    second.name = "shell";
    second.A = 12.0;
    second.Z = 6.0;
    cfg.materials.materials.push_back(second);
    REQUIRE_THROWS_WITH(
        tenryu::core::validate_ale1d_config(cfg),
        Catch::Matchers::ContainsSubstring("exactly one material"));
  }
  {
    auto cfg = make_ale();
    cfg.materials.materials.front().eos_model = "tmat";
    REQUIRE_THROWS_WITH(
        tenryu::core::validate_ale1d_config(cfg),
        Catch::Matchers::ContainsSubstring("ideal_gas"));
  }
  {
    auto cfg = make_ale();
    cfg.burn.enabled = true;
    REQUIRE_THROWS_WITH(
        tenryu::core::validate_ale1d_config(cfg),
        Catch::Matchers::ContainsSubstring("Burn.enabled"));
  }
  {
    auto cfg = make_ale();
    cfg.radiation.enabled = true;
    cfg.radiation.mode = tenryu::core::RadiationMode::SnTransport;
    REQUIRE_THROWS_WITH(
        tenryu::core::validate_ale1d_config(cfg),
        Catch::Matchers::ContainsSubstring("sn_transport"));
  }
}

// ---- tests/core/test_namelist_ai_review_1d_keys.cpp, lines 305-371 at 6d62bf929 ----
TEST_CASE("ALE1D trigger threshold and candidate gate keys validate and freeze",
          "[core][namelist][ale1d]") {
  auto cfg = make_1d_config();
  cfg.numerics.ale1d.enabled = true;
  REQUIRE_NOTHROW(tenryu::core::validate_ale1d_config(cfg));
  {
    auto bad = cfg;
    bad.numerics.ale1d.emergency_max_dr_ratio = 0.9;
    REQUIRE_THROWS_WITH(tenryu::core::validate_ale1d_config(bad),
                        Catch::Matchers::ContainsSubstring("emergency_max_dr_ratio"));
  }
  {
    auto bad = cfg;
    bad.numerics.ale1d.candidate_dt_penalty_max = 0.5;
    REQUIRE_THROWS_WITH(tenryu::core::validate_ale1d_config(bad),
                        Catch::Matchers::ContainsSubstring("candidate_dt_penalty_max"));
  }
  {
    auto bad = cfg;
    bad.numerics.ale1d.benefit_min_dt_gain = 0.5;
    REQUIRE_THROWS_WITH(tenryu::core::validate_ale1d_config(bad),
                        Catch::Matchers::ContainsSubstring("benefit_min_dt_gain"));
  }
#if defined(TENRYU_ENABLE_PYTHON) && TENRYU_ENABLE_PYTHON
  ensure_python_started();
  py::gil_scoped_acquire gil;
  const std::string current = tenryu::core::namelist::Freeze::to_checkpoint_json(cfg);
  py::module_ json = py::module_::import("json");
  py::dict root = json.attr("loads")(current).cast<py::dict>();
  py::dict numerics = root[py::str("numerics")].cast<py::dict>();
  py::dict ale1d = numerics[py::str("ale1d")].cast<py::dict>();
  REQUIRE(ale1d.contains(py::str("emergency_max_dr_ratio")));
  CHECK(ale1d[py::str("emergency_max_dr_ratio")].cast<double>() == 1.25);
  py::dict floor = ale1d[py::str("min_width_floor")].cast<py::dict>();
  REQUIRE(floor.contains(py::str("retrigger_cooldown_steps")));

  // A checkpoint written before these keys existed compares equal (the
  // restart asserts frozen-config equivalence); each key separately.
  const auto without = [&](const std::function<void(py::dict&)>& drop) {
    py::dict copy = json.attr("loads")(current).cast<py::dict>();
    drop(copy);
    return json.attr("dumps")(copy, py::arg("sort_keys") = true).cast<std::string>();
  };
  const auto ale1d_of = [](py::dict& root_dict) {
    return root_dict[py::str("numerics")].cast<py::dict>()[py::str("ale1d")].cast<py::dict>();
  };
  CHECK(tenryu::core::namelist::Freeze::configs_equivalent(
      current, without([&](py::dict& r) {
        ale1d_of(r).attr("pop")(py::str("emergency_max_dr_ratio"));
      })));
  CHECK(tenryu::core::namelist::Freeze::configs_equivalent(
      current, without([&](py::dict& r) {
        ale1d_of(r)[py::str("min_width_floor")].cast<py::dict>().attr("pop")(
            py::str("retrigger_cooldown_steps"));
      })));
  CHECK(tenryu::core::namelist::Freeze::configs_equivalent(
      current, without([&](py::dict& r) {
        ale1d_of(r).attr("pop")(py::str("min_width_floor"));
      })));

  // A non-default threshold is a different configuration.
  auto changed = cfg;
  changed.numerics.ale1d.emergency_max_dr_ratio = 1.7;
  CHECK_FALSE(tenryu::core::namelist::Freeze::configs_equivalent(
      current, tenryu::core::namelist::Freeze::to_checkpoint_json(changed)));
#endif
}

// ---- tests/core/test_namelist_radiation_guards.cpp, lines 111-114 at 6d62bf929 ----
  SECTION("ALE1D is outside this mode's scope") {
    cfg.numerics.ale1d.enabled = true;
    require_config_error_contains(cfg, "conservative_advection requires");
  }
