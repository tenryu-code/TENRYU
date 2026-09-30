// Retired with the Monte Carlo radiation on 2026-09-29: the test case of tests/io/test_hdf5_roundtrip.cpp as it was at
// 5bc8f6ce3, which also round-tripped a photon pool with IMC and DDMC particles and the HOLO state through a
// checkpoint. The current test case keeps its snapshot/checkpoint fields without them. Not built.

TEST_CASE("HDF5 snapshot/checkpoint roundtrip", "[io][hdf5]") {
#if TENRYU_ENABLE_HDF5
  auto cfg = make_config();
  std::filesystem::remove_all(cfg.output.directory);
  std::filesystem::create_directories(cfg.output.directory);

  auto state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;

  std::vector<double> rho(state.rho.size(), 0.0);
  std::vector<double> Te(state.Te.size(), 0.0);
  std::vector<double> Ti(state.Ti.size(), 0.0);
  std::vector<double> ee(state.ee.size(), 0.0);
  std::vector<double> mass(state.mass.size(), 0.0);
  std::vector<double> zbar(state.zbar.size(), 1.0);
  std::vector<double> rad_E(state.rad_E.size(), 0.0);
  std::vector<double> rad_dep(state.rad_dep.size(), 0.0);
  std::vector<double> rad_emit(state.rad_emit.size(), 0.0);
  std::vector<double> holo_E_LO(state.holo_E_LO.size(), 0.0);
  std::vector<double> holo_consistency_source(
      state.holo_consistency_source.size(), 0.0);
  std::vector<double> holo_rad_dep(state.holo_rad_dep.size(), 0.0);
  std::vector<double> holo_rad_emit(state.holo_rad_emit.size(), 0.0);
  std::vector<double> holo_Prr(state.holo_Prr.size(), 0.0);
  std::vector<double> holo_chi(state.holo_chi.size(), 0.0);
  std::vector<double> holo_Prr_coverage(state.holo_Prr_coverage.size(), 0.0);
  std::vector<double> laser_dep(state.laser_dep.size(), 0.0);

  for (std::size_t i = 0; i < rho.size(); ++i) {
    rho[i] = 1.0 + 0.1 * static_cast<double>(i);
    Te[i] = 2.0 + 0.01 * static_cast<double>(i);
    Ti[i] = 3.0 + 0.02 * static_cast<double>(i);
    ee[i] = 4.0 + 0.03 * static_cast<double>(i);
    mass[i] = rho[i] * state.mesh.cell_vol[i];
    laser_dep[i] = 5.0 + 0.05 * static_cast<double>(i);
  }
  for (std::size_t i = 0; i < rad_E.size(); ++i) {
    rad_E[i] = 10.0 + 0.1 * static_cast<double>(i);
    rad_dep[i] = 0.5 + 0.01 * static_cast<double>(i);
    rad_emit[i] = 0.25 + 0.02 * static_cast<double>(i);
    holo_E_LO[i] = 20.0 + 0.2 * static_cast<double>(i);
    holo_consistency_source[i] = 1.5e12 + 7.0e10 * static_cast<double>(i);
    holo_rad_dep[i] = 0.75 + 0.03 * static_cast<double>(i);
    holo_rad_emit[i] = 0.35 + 0.04 * static_cast<double>(i);
    holo_Prr[i] = 0.15 + 0.01 * static_cast<double>(i);
    holo_chi[i] = 0.25 + 0.005 * static_cast<double>(i);
    holo_Prr_coverage[i] = 0.8 - 0.01 * static_cast<double>(i);
  }

  state.rho.copy_from_host(rho.data());
  state.Te.copy_from_host(Te.data());
  state.Ti.copy_from_host(Ti.data());
  state.ee.copy_from_host(ee.data());
  state.mass.copy_from_host(mass.data());
  state.zbar.copy_from_host(zbar.data());
  state.rad_E.copy_from_host(rad_E.data());
  state.rad_dep.copy_from_host(rad_dep.data());
  state.rad_emit.copy_from_host(rad_emit.data());
  state.holo_E_LO.copy_from_host(holo_E_LO.data());
  state.holo_consistency_source.copy_from_host(holo_consistency_source.data());
  state.holo_rad_dep.copy_from_host(holo_rad_dep.data());
  state.holo_rad_emit.copy_from_host(holo_rad_emit.data());
  state.holo_Prr.copy_from_host(holo_Prr.data());
  state.holo_chi.copy_from_host(holo_chi.data());
  state.holo_Prr_coverage.copy_from_host(holo_Prr_coverage.data());
  state.laser_dep.copy_from_host(laser_dep.data());
  state.holo_core_mask = {0, 1, 1, 0, 0, 0, 0, 0};
  state.holo_core_prev_mask = {0, 0, 1, 1, 0, 0, 0, 0};
  state.holo_hold_count = {0, 2, 0, 0, 1, 0, 0, 0};
  state.holo_dwell_count = {0, 5, 6, 0, 0, 0, 0, 0};
  state.holo_tau_R = {0.0, 5.0, 6.0, 2.0, 0.0, 0.0, 0.0, 0.0};
  state.holo_reduced_flux = {0.0, 0.1, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0};
  state.holo_mass_q = {0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75};
  state.holo_core_mask_valid = true;

  state.t = 2.5e-10;
  state.step = 12;
  state.dt = 1.0e-11;
  state.E_laser_deposited = 1.2;
  state.E_laser_escaped = 0.3;
  state.E_laser_incident = 0.7;
  state.E_ra_deposited = 0.11;
  state.E_rad_escaped = 0.2;
  state.E_numerical_loss = 0.01;
  state.E_floor_injected = 0.02;
  state.E_pdV_bdry = 0.03;
  state.E_Marshak_in = 0.04;
  state.t_next_plot = 3.0e-10;
  state.t_next_history = 2.6e-10;
  state.t_next_checkpoint = 4.0e-10;

  tenryu::radiation::PhotonPool pool;
  pool.allocate(8);
  pool.n_alive = 3;
  pool.n_census = 3;

  std::vector<double> pos_r = {0.1, 0.2, 0.3};
  std::vector<double> pos_z = {0.0, 0.0, 0.0};
  std::vector<double> dir_r = {1.0, 0.0, 0.0};
  std::vector<double> dir_z = {0.0, 1.0, 0.0};
  std::vector<double> dir_phi = {0.0, 0.0, 1.0};
  std::vector<double> energy = {1.0, 2.0, 3.0};
  std::vector<double> birth_energy = {1.1, 2.1, 3.1};
  std::vector<double> weight = {1.0, 1.0, 1.0};
  std::vector<double> time_remain = {1.0e-12, 2.0e-12, 3.0e-12};
  std::vector<std::uint64_t> global_id = {11, 22, 33};
  std::vector<std::uint32_t> rng_counter = {3, 4, 5};
  std::vector<std::int32_t> cell_id = {0, 1, 2};
  std::vector<std::uint16_t> group_id = {0, 1, 0};
  std::vector<std::uint8_t> mode = {
      tenryu::radiation::kModeIMC,
      tenryu::radiation::kModeDDMC,
      tenryu::radiation::kModeIMC,
  };
  std::vector<std::uint8_t> alive = {
      tenryu::radiation::kAlive,
      tenryu::radiation::kAlive,
      tenryu::radiation::kAlive,
  };

  cuda_check(cudaMemcpy(pool.pos_r, pos_r.data(), sizeof(double) * pos_r.size(), cudaMemcpyHostToDevice), "pool pos_r copy failed");
  cuda_check(cudaMemcpy(pool.pos_z, pos_z.data(), sizeof(double) * pos_z.size(), cudaMemcpyHostToDevice), "pool pos_z copy failed");
  cuda_check(cudaMemcpy(pool.dir_r, dir_r.data(), sizeof(double) * dir_r.size(), cudaMemcpyHostToDevice), "pool dir_r copy failed");
  cuda_check(cudaMemcpy(pool.dir_z, dir_z.data(), sizeof(double) * dir_z.size(), cudaMemcpyHostToDevice), "pool dir_z copy failed");
  cuda_check(cudaMemcpy(pool.dir_phi, dir_phi.data(), sizeof(double) * dir_phi.size(), cudaMemcpyHostToDevice), "pool dir_phi copy failed");
  cuda_check(cudaMemcpy(pool.energy, energy.data(), sizeof(double) * energy.size(), cudaMemcpyHostToDevice), "pool energy copy failed");
  cuda_check(cudaMemcpy(pool.birth_energy, birth_energy.data(), sizeof(double) * birth_energy.size(), cudaMemcpyHostToDevice), "pool birth_energy copy failed");
  cuda_check(cudaMemcpy(pool.weight, weight.data(), sizeof(double) * weight.size(), cudaMemcpyHostToDevice), "pool weight copy failed");
  cuda_check(cudaMemcpy(pool.time_remain, time_remain.data(), sizeof(double) * time_remain.size(), cudaMemcpyHostToDevice), "pool time_remain copy failed");
  cuda_check(cudaMemcpy(pool.global_id, global_id.data(), sizeof(std::uint64_t) * global_id.size(), cudaMemcpyHostToDevice), "pool global_id copy failed");
  cuda_check(cudaMemcpy(pool.rng_counter, rng_counter.data(), sizeof(std::uint32_t) * rng_counter.size(), cudaMemcpyHostToDevice), "pool rng_counter copy failed");
  cuda_check(cudaMemcpy(pool.cell_id, cell_id.data(), sizeof(std::int32_t) * cell_id.size(), cudaMemcpyHostToDevice), "pool cell_id copy failed");
  cuda_check(cudaMemcpy(pool.group_id, group_id.data(), sizeof(std::uint16_t) * group_id.size(), cudaMemcpyHostToDevice), "pool group_id copy failed");
  cuda_check(cudaMemcpy(pool.mode, mode.data(), sizeof(std::uint8_t) * mode.size(), cudaMemcpyHostToDevice), "pool mode copy failed");
  cuda_check(cudaMemcpy(pool.alive, alive.data(), sizeof(std::uint8_t) * alive.size(), cudaMemcpyHostToDevice), "pool alive copy failed");

  tenryu::io::HDF5Writer writer;
  writer.write_snapshot(state, cfg, /*file_index=*/0, state.step, state.t, cfg.output.directory, cfg.main.name);
  writer.write_checkpoint(state, cfg, pool, /*file_index=*/0, state.step, state.t, cfg.output.directory, cfg.main.name);

  const std::filesystem::path snapshot =
      std::filesystem::path(cfg.output.directory) / "roundtrip_0000.h5";
  const std::filesystem::path checkpoint =
      std::filesystem::path(cfg.output.directory) / "roundtrip_ckpt_0000.h5";
  REQUIRE(std::filesystem::exists(snapshot));
  REQUIRE(std::filesystem::exists(checkpoint));

  tenryu::io::HDF5Reader reader;
  auto restored = reader.read_checkpoint(
      cfg,
      (std::filesystem::path(cfg.output.directory) / "roundtrip_ckpt_0000").string());

  const auto rho_restored = to_host(restored.state.rho);
  REQUIRE(rho_restored.size() == rho.size());
  for (std::size_t i = 0; i < rho.size(); ++i) {
    REQUIRE(rho_restored[i] == Catch::Approx(rho[i]).epsilon(1.0e-12));
  }

  REQUIRE(restored.state.step == state.step);
  REQUIRE(restored.state.t == Catch::Approx(state.t).epsilon(1.0e-12));
  REQUIRE(restored.state.dt == Catch::Approx(state.dt).epsilon(1.0e-12));
  REQUIRE(restored.state.E_laser_deposited ==
          Catch::Approx(state.E_laser_deposited).epsilon(1.0e-12));
  REQUIRE(restored.state.E_laser_escaped ==
          Catch::Approx(state.E_laser_escaped).epsilon(1.0e-12));
  REQUIRE(restored.state.E_laser_incident ==
          Catch::Approx(state.E_laser_incident).epsilon(1.0e-12));
  REQUIRE(restored.state.E_ra_deposited ==
          Catch::Approx(state.E_ra_deposited).epsilon(1.0e-12));
  REQUIRE(restored.photon_pool.n_alive == pool.n_alive);
  REQUIRE(restored.state.holo_core_mask_valid);
  REQUIRE(restored.state.holo_core_mask == state.holo_core_mask);
  REQUIRE(restored.state.holo_core_prev_mask == state.holo_core_prev_mask);
  REQUIRE(restored.state.holo_hold_count == state.holo_hold_count);
  REQUIRE(restored.state.holo_dwell_count == state.holo_dwell_count);
  REQUIRE(restored.state.holo_tau_R == state.holo_tau_R);
  REQUIRE(restored.state.holo_reduced_flux == state.holo_reduced_flux);
  REQUIRE(restored.state.holo_mass_q == state.holo_mass_q);
  const auto holo_E_LO_restored = to_host(restored.state.holo_E_LO);
  const auto holo_consistency_source_restored =
      to_host(restored.state.holo_consistency_source);
  const auto rad_emit_restored = to_host(restored.state.rad_emit);
  const auto holo_rad_dep_restored = to_host(restored.state.holo_rad_dep);
  const auto holo_rad_emit_restored = to_host(restored.state.holo_rad_emit);
  const auto holo_Prr_restored = to_host(restored.state.holo_Prr);
  const auto holo_chi_restored = to_host(restored.state.holo_chi);
  const auto holo_Prr_coverage_restored = to_host(restored.state.holo_Prr_coverage);
  REQUIRE(rad_emit_restored.size() == rad_emit.size());
  REQUIRE(holo_rad_dep_restored.size() == holo_rad_dep.size());
  REQUIRE(holo_rad_emit_restored.size() == holo_rad_emit.size());
  REQUIRE(holo_E_LO_restored.size() == holo_E_LO.size());
  REQUIRE(holo_consistency_source_restored.size() == holo_consistency_source.size());
  REQUIRE(holo_Prr_restored.size() == holo_Prr.size());
  REQUIRE(holo_chi_restored.size() == holo_chi.size());
  REQUIRE(holo_Prr_coverage_restored.size() == holo_Prr_coverage.size());
  for (std::size_t i = 0; i < holo_E_LO.size(); ++i) {
    REQUIRE(rad_emit_restored[i] == Catch::Approx(rad_emit[i]).epsilon(1.0e-12));
    REQUIRE(holo_E_LO_restored[i] == Catch::Approx(holo_E_LO[i]).epsilon(1.0e-12));
    REQUIRE(holo_consistency_source_restored[i] ==
            Catch::Approx(holo_consistency_source[i]).epsilon(1.0e-12));
    REQUIRE(holo_rad_dep_restored[i] == Catch::Approx(holo_rad_dep[i]).epsilon(1.0e-12));
    REQUIRE(holo_rad_emit_restored[i] == Catch::Approx(holo_rad_emit[i]).epsilon(1.0e-12));
    REQUIRE(holo_Prr_restored[i] == Catch::Approx(holo_Prr[i]).epsilon(1.0e-12));
    REQUIRE(holo_chi_restored[i] == Catch::Approx(holo_chi[i]).epsilon(1.0e-12));
    REQUIRE(holo_Prr_coverage_restored[i] ==
            Catch::Approx(holo_Prr_coverage[i]).epsilon(1.0e-12));
  }

  std::vector<double> energy_restored(static_cast<std::size_t>(restored.photon_pool.n_alive), 0.0);
  cuda_check(cudaMemcpy(energy_restored.data(),
                        restored.photon_pool.energy,
                        sizeof(double) * energy_restored.size(),
                        cudaMemcpyDeviceToHost),
             "restored pool energy copy failed");
  REQUIRE(energy_restored[0] == Catch::Approx(1.0));
  REQUIRE(energy_restored[1] == Catch::Approx(2.0));
  REQUIRE(energy_restored[2] == Catch::Approx(3.0));
#else
  SUCCEED("TENRYU_ENABLE_HDF5=OFF");
#endif
}
