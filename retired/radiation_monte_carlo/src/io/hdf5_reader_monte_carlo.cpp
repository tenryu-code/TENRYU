// Retired with the Monte Carlo radiation on 2026-09-29: the checkpoint restore of the photon pool, the DDMC mode map
// and the HOLO state, cut out of src/io/hdf5_reader.cpp as it was at 5bc8f6ce3. Not built.

// ---- restore_particles

void restore_particles(const hid_t file,
                       radiation::PhotonPool& pool,
                       std::vector<std::int8_t>* ddmc_mode_map,
                       const std::size_t n_cells,
                       const std::size_t n_groups) {
  if (ddmc_mode_map != nullptr) {
    ddmc_mode_map->assign(checked_mul_size(n_cells, n_groups, "ddmc mode-map size"), 0);
  }
  if (!link_exists(file, "particles")) {
    return;
  }

  const auto n_particles_attr = read_group_attr_i64(file, "particles", "n_particles");
  if (n_particles_attr.has_value()) {
    TENRYU_ASSERT(*n_particles_attr >= 0,
                  "Restart checkpoint has negative particles/n_particles attribute");
  }

  const bool has_energy_dataset = link_exists(file, "particles/energy");
  const std::size_t energy_dataset_size =
      has_energy_dataset ? dataset_size(file, "particles/energy") : 0;

  const std::size_t n_p =
      n_particles_attr.has_value() ? checked_i64_to_size(*n_particles_attr, "particles/n_particles")
                                   : energy_dataset_size;
  if (n_particles_attr.has_value() && has_energy_dataset) {
    TENRYU_ASSERT(
        energy_dataset_size == n_p,
        "Restart checkpoint particle count mismatch: particles/n_particles=" +
            std::to_string(n_p) + ", particles/energy size=" + std::to_string(energy_dataset_size));
  }

  TENRYU_ASSERT(n_p <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "Restart checkpoint particle count exceeds INT_MAX");
  const std::int64_t pool_capacity_attr =
      read_scalar_dataset<std::int64_t>(file, "particles/pool_capacity", H5T_NATIVE_INT64, 0);
  const std::size_t pool_capacity_hint =
      (pool_capacity_attr > 0) ? checked_i64_to_size(pool_capacity_attr, "particles/pool_capacity")
                               : 0;
  const std::size_t pool_capacity_size = std::max(n_p, pool_capacity_hint);
  const int pool_capacity =
      checked_size_to_int(pool_capacity_size, "Restart checkpoint particles/pool_capacity");

  if (pool_capacity <= 0) {
    return;
  }

  pool.allocate(pool_capacity);

  if (n_p == 0) {
    pool.n_alive = 0;
    pool.n_census = 0;
    return;
  }

  auto pos_r = read_vector_dataset_checked<double>(file, "particles/pos_r", H5T_NATIVE_DOUBLE, n_p);
  auto pos_z = read_vector_dataset_checked<double>(file, "particles/pos_z", H5T_NATIVE_DOUBLE, n_p);
  auto dir_r = read_vector_dataset_checked<double>(file, "particles/dir_r", H5T_NATIVE_DOUBLE, n_p);
  auto dir_z = read_vector_dataset_checked<double>(file, "particles/dir_z", H5T_NATIVE_DOUBLE, n_p);
  auto dir_phi =
      read_vector_dataset_checked<double>(file, "particles/dir_phi", H5T_NATIVE_DOUBLE, n_p);
  auto energy =
      read_vector_dataset_checked<double>(file, "particles/energy", H5T_NATIVE_DOUBLE, n_p);
  auto birth_energy =
      read_vector_dataset_checked<double>(file, "particles/birth_energy", H5T_NATIVE_DOUBLE, n_p);
  const bool has_sign = link_exists(file, "particles/sign");
  auto sign = has_sign
                  ? read_vector_dataset_checked<std::int8_t>(
                        file, "particles/sign", H5T_NATIVE_INT8, n_p)
                  : std::vector<std::int8_t>(n_p, 1);
  if (!has_sign) {
    core::log_warning(
        "Restart checkpoint missing particles/sign; defaulting all particle signs to +1.");
  }
  auto group_id =
      read_vector_dataset_checked<std::uint16_t>(file, "particles/group_id", H5T_NATIVE_UINT16, n_p);
  auto cell_id =
      read_vector_dataset_checked<std::int32_t>(file, "particles/cell_id", H5T_NATIVE_INT32, n_p);
  auto mode =
      read_vector_dataset_checked<std::uint8_t>(file, "particles/mode", H5T_NATIVE_UINT8, n_p);
  auto global_id = read_vector_dataset_checked<std::uint64_t>(
      file, "particles/global_id", H5T_NATIVE_UINT64, n_p);
  auto weight =
      read_vector_dataset_checked<double>(file, "particles/weight", H5T_NATIVE_DOUBLE, n_p);
  auto time_remain =
      read_vector_dataset_checked<double>(file, "particles/time_remain", H5T_NATIVE_DOUBLE, n_p);

  const bool has_rng_counter = link_exists(file, "rng/rng_counter");
  auto rng_counter = has_rng_counter
                         ? read_vector_dataset_checked<std::uint32_t>(
                               file, "rng/rng_counter", H5T_NATIVE_UINT32, n_p)
                         : std::vector<std::uint32_t>(n_p, 0u);
  if (!has_rng_counter) {
    core::log_warning(
        "Restart checkpoint missing rng/rng_counter; defaulting counters to zero "
        "(reproducibility may change).");
  }
  if (link_exists(file, "rng/global_id")) {
    global_id =
        read_vector_dataset_checked<std::uint64_t>(file, "rng/global_id", H5T_NATIVE_UINT64, n_p);
  }
  const bool has_alive = link_exists(file, "particles/alive");
  auto alive = has_alive
                   ? read_vector_dataset_checked<std::uint8_t>(
                         file, "particles/alive", H5T_NATIVE_UINT8, n_p)
                   : std::vector<std::uint8_t>(n_p, radiation::kAlive);
  if (!has_alive) {
    core::log_warning(
        "Restart checkpoint missing particles/alive; defaulting all particles to alive=1.");
  }

  const std::size_t n = n_p;
  pos_r.resize(n, std::numeric_limits<double>::quiet_NaN());
  pos_z.resize(n, std::numeric_limits<double>::quiet_NaN());
  dir_r.resize(n, std::numeric_limits<double>::quiet_NaN());
  dir_z.resize(n, std::numeric_limits<double>::quiet_NaN());
  dir_phi.resize(n, std::numeric_limits<double>::quiet_NaN());
  group_id.resize(n, 0);
  cell_id.resize(n, 0);
  mode.resize(n, radiation::kModeIMC);
  global_id.resize(n, 0);
  energy.resize(n, 0.0);
  birth_energy.resize(n, 0.0);
  sign.resize(n, 1);
  weight.resize(n, 1.0);
  time_remain.resize(n, 0.0);
  rng_counter.resize(n, 0);
  alive.resize(n, radiation::kAlive);

  std::size_t invalid_group_id = 0;
  std::size_t invalid_cell_id = 0;
  std::size_t invalid_energy = 0;
  std::size_t invalid_sign = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (sign[i] != -1 && sign[i] != 1) {
      sign[i] = 1;
      ++invalid_sign;
    }
    const bool bad_group = static_cast<std::size_t>(group_id[i]) >= n_groups;
    const bool bad_cell = (cell_id[i] < 0) || (static_cast<std::size_t>(cell_id[i]) >= n_cells);
    const bool bad_energy = energy[i] < 0.0;
    if (!(bad_group || bad_cell || bad_energy)) {
      continue;
    }
    alive[i] = radiation::kDead;
    if (bad_group) {
      group_id[i] = 0;
      ++invalid_group_id;
    }
    if (bad_cell) {
      cell_id[i] = 0;
      ++invalid_cell_id;
    }
    if (bad_energy) {
      energy[i] = 0.0;
      birth_energy[i] = std::max(birth_energy[i], 0.0);
      ++invalid_energy;
    }
  }
  if (invalid_group_id > 0 || invalid_cell_id > 0 || invalid_energy > 0) {
    core::log_warning("Restart: invalid particle fields detected; marked particles dead "
                      "(invalid group_id=" + std::to_string(invalid_group_id) +
                      ", invalid cell_id=" + std::to_string(invalid_cell_id) +
                      ", negative energy=" + std::to_string(invalid_energy) + ").");
  }
  if (invalid_sign > 0) {
    core::log_warning("Restart: invalid particle signs detected; reset to +1 "
                      "(invalid sign=" + std::to_string(invalid_sign) + ").");
  }

  if (ddmc_mode_map != nullptr && !ddmc_mode_map->empty()) {
    for (std::size_t i = 0; i < n; ++i) {
      if (alive[i] != radiation::kAlive) {
        continue;
      }
      if (mode[i] != radiation::kModeDDMC && mode[i] != radiation::kModeRW) {
        continue;
      }
      const int cell = cell_id[i];
      const int group = static_cast<int>(group_id[i]);
      if (cell < 0 || static_cast<std::size_t>(cell) >= n_cells || group < 0 ||
          static_cast<std::size_t>(group) >= n_groups) {
        continue;
      }
      const std::size_t idx =
          static_cast<std::size_t>(cell) * n_groups + static_cast<std::size_t>(group);
      (*ddmc_mode_map)[idx] = static_cast<std::int8_t>(mode[i]);
    }
  }

  bool warned_ddmc_nan = false;
  for (std::size_t i = 0; i < n; ++i) {
    if (mode[i] != radiation::kModeDDMC) {
      continue;
    }
    const bool is_nan = std::isnan(pos_r[i]) && std::isnan(pos_z[i]) && std::isnan(dir_r[i]) &&
                        std::isnan(dir_z[i]) && std::isnan(dir_phi[i]);
    if (!is_nan) {
      pos_r[i] = std::numeric_limits<double>::quiet_NaN();
      pos_z[i] = std::numeric_limits<double>::quiet_NaN();
      dir_r[i] = std::numeric_limits<double>::quiet_NaN();
      dir_z[i] = std::numeric_limits<double>::quiet_NaN();
      dir_phi[i] = std::numeric_limits<double>::quiet_NaN();
      warned_ddmc_nan = true;
    }
  }
  if (warned_ddmc_nan) {
    core::log_warning("Restart: repaired DDMC NaN sentinel on legacy checkpoint particles");
  }

  const std::size_t bytes_double =
      checked_bytes_for_count<double>(n, "restart particle double-array copy");
  const std::size_t bytes_group_id =
      checked_bytes_for_count<std::uint16_t>(n, "restart particles/group_id copy");
  const std::size_t bytes_cell_id =
      checked_bytes_for_count<std::int32_t>(n, "restart particles/cell_id copy");
  const std::size_t bytes_sign =
      checked_bytes_for_count<std::int8_t>(n, "restart particles/sign copy");
  const std::size_t bytes_mode =
      checked_bytes_for_count<std::uint8_t>(n, "restart particles/mode copy");
  const std::size_t bytes_global_id =
      checked_bytes_for_count<std::uint64_t>(n, "restart particles/global_id copy");
  const std::size_t bytes_rng_counter =
      checked_bytes_for_count<std::uint32_t>(n, "restart rng/rng_counter copy");

  cuda_check(cudaMemcpy(pool.pos_r, pos_r.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy pos_r failed");
  cuda_check(cudaMemcpy(pool.pos_z, pos_z.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy pos_z failed");
  cuda_check(cudaMemcpy(pool.dir_r, dir_r.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy dir_r failed");
  cuda_check(cudaMemcpy(pool.dir_z, dir_z.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy dir_z failed");
  cuda_check(cudaMemcpy(pool.dir_phi, dir_phi.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy dir_phi failed");
  cuda_check(cudaMemcpy(pool.energy, energy.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy energy failed");
  cuda_check(cudaMemcpy(pool.birth_energy, birth_energy.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy birth_energy failed");
  cuda_check(cudaMemcpy(pool.sign, sign.data(), bytes_sign, cudaMemcpyHostToDevice),
             "restart copy sign failed");
  cuda_check(cudaMemcpy(pool.group_id, group_id.data(), bytes_group_id, cudaMemcpyHostToDevice),
             "restart copy group_id failed");
  cuda_check(cudaMemcpy(pool.cell_id, cell_id.data(), bytes_cell_id, cudaMemcpyHostToDevice),
             "restart copy cell_id failed");
  cuda_check(cudaMemcpy(pool.mode, mode.data(), bytes_mode, cudaMemcpyHostToDevice),
             "restart copy mode failed");
  cuda_check(cudaMemcpy(pool.global_id, global_id.data(), bytes_global_id, cudaMemcpyHostToDevice),
             "restart copy global_id failed");
  cuda_check(cudaMemcpy(pool.weight, weight.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy weight failed");
  cuda_check(cudaMemcpy(pool.time_remain, time_remain.data(), bytes_double, cudaMemcpyHostToDevice),
             "restart copy time_remain failed");
  cuda_check(cudaMemcpy(pool.rng_counter,
                        rng_counter.data(),
                        bytes_rng_counter,
                        cudaMemcpyHostToDevice),
             "restart copy rng_counter failed");

  cuda_check(cudaMemcpy(pool.alive, alive.data(), bytes_mode, cudaMemcpyHostToDevice),
             "restart copy alive failed");

  pool.n_alive = checked_size_to_int(n, "Restart checkpoint particle count");
  pool.n_census = pool.n_alive;
}

// ---- radiation source memory, DDMC flag and HOLO restore

  if (link_exists(file, "radiation/delta_E_rad_prev")) {
    auto delta_E_rad_prev =
        read_vector_dataset<double>(file, "radiation/delta_E_rad_prev", H5T_NATIVE_DOUBLE);
    if (delta_E_rad_prev.size() != out.state.rho.size()) {
      core::log_warning("Restart checkpoint dataset size mismatch for radiation/delta_E_rad_prev "
                        "(checkpoint=" +
                        std::to_string(delta_E_rad_prev.size()) + ", state=" +
                        std::to_string(out.state.rho.size()) +
                        "); truncating/padding with zeros.");
    }
    if (!delta_E_rad_prev.empty()) {
      delta_E_rad_prev.resize(out.state.rho.size(), 0.0);
      out.state.delta_E_rad_prev.reset(out.state.rho.size());
      out.state.delta_E_rad_prev.copy_from_host(delta_E_rad_prev.data());
    } else {
      out.state.delta_E_rad_prev.reset(0);
    }
  } else {
    out.state.delta_E_rad_prev.reset(0);
  }
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups =
      checked_mul_size(out.state.rho.size(), n_groups, "restart radiation/ddmc_flag size");
  bool has_checkpoint_ddmc_map = false;
  if (link_exists(file, "radiation/ddmc_flag")) {
    out.state.ddmc_mode_map = read_vector_dataset_checked<std::int8_t>(
        file, "radiation/ddmc_flag", H5T_NATIVE_INT8, n_cell_groups);
    has_checkpoint_ddmc_map = true;
    out.state.ddmc_mode_map_valid = true;
  } else {
    out.state.ddmc_mode_map.assign(n_cell_groups, static_cast<std::int8_t>(0));
    out.state.ddmc_mode_map_valid = false;
  }

  const std::size_t n_cells = out.state.rho.size();
  if (link_exists(file, "holo/E_LO")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/E_LO", H5T_NATIVE_DOUBLE),
                         out.state.holo_E_LO,
                         "holo/E_LO");
  } else {
    out.state.holo_E_LO.fill(0.0);
  }
  if (link_exists(file, "holo/consistency_source")) {
    copy_vector_to_field(
        read_vector_dataset<double>(file, "holo/consistency_source", H5T_NATIVE_DOUBLE),
        out.state.holo_consistency_source,
        "holo/consistency_source");
  } else {
    out.state.holo_consistency_source.fill(0.0);
  }
  if (link_exists(file, "holo/rad_dep_LO")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/rad_dep_LO", H5T_NATIVE_DOUBLE),
                         out.state.holo_rad_dep,
                         "holo/rad_dep_LO");
  } else {
    out.state.holo_rad_dep.fill(0.0);
  }
  if (link_exists(file, "holo/rad_emit_LO")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/rad_emit_LO", H5T_NATIVE_DOUBLE),
                         out.state.holo_rad_emit,
                         "holo/rad_emit_LO");
  } else {
    out.state.holo_rad_emit.fill(0.0);
  }
  if (link_exists(file, "holo/Prr_HO")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/Prr_HO", H5T_NATIVE_DOUBLE),
                         out.state.holo_Prr,
                         "holo/Prr_HO");
  } else {
    out.state.holo_Prr.fill(0.0);
  }
  if (link_exists(file, "holo/chi")) {
    copy_vector_to_field(read_vector_dataset<double>(file, "holo/chi", H5T_NATIVE_DOUBLE),
                         out.state.holo_chi,
                         "holo/chi");
  } else {
    out.state.holo_chi.fill(0.0);
  }
  if (link_exists(file, "holo/Prr_coverage")) {
    copy_vector_to_field(
        read_vector_dataset<double>(file, "holo/Prr_coverage", H5T_NATIVE_DOUBLE),
        out.state.holo_Prr_coverage,
        "holo/Prr_coverage");
  } else {
    out.state.holo_Prr_coverage.fill(0.0);
  }
  if (link_exists(file, "holo/core_mask")) {
    out.state.holo_core_mask = read_vector_dataset_checked<std::uint8_t>(
        file, "holo/core_mask", H5T_NATIVE_UINT8, n_cells);
    out.state.holo_core_mask_valid = true;
  } else {
    out.state.holo_core_mask.assign(n_cells, static_cast<std::uint8_t>(0));
    out.state.holo_core_mask_valid = false;
  }
  if (link_exists(file, "holo/prev_core_mask")) {
    out.state.holo_core_prev_mask = read_vector_dataset_checked<std::uint8_t>(
        file, "holo/prev_core_mask", H5T_NATIVE_UINT8, n_cells);
  } else {
    out.state.holo_core_prev_mask = out.state.holo_core_mask;
  }
  if (link_exists(file, "holo/hold_count")) {
    out.state.holo_hold_count = read_vector_dataset_checked<std::int32_t>(
        file, "holo/hold_count", H5T_NATIVE_INT32, n_cells);
  } else {
    out.state.holo_hold_count.assign(n_cells, 0);
  }
  if (link_exists(file, "holo/dwell_count")) {
    out.state.holo_dwell_count = read_vector_dataset_checked<std::int32_t>(
        file, "holo/dwell_count", H5T_NATIVE_INT32, n_cells);
  } else {
    out.state.holo_dwell_count.assign(n_cells, 0);
  }
  if (link_exists(file, "holo/tau_R")) {
    out.state.holo_tau_R = read_vector_dataset_checked<double>(
        file, "holo/tau_R", H5T_NATIVE_DOUBLE, n_cells);
  } else {
    out.state.holo_tau_R.assign(n_cells, 0.0);
  }
  if (link_exists(file, "holo/reduced_flux")) {
    out.state.holo_reduced_flux = read_vector_dataset_checked<double>(
        file, "holo/reduced_flux", H5T_NATIVE_DOUBLE, n_cells);
  } else {
    out.state.holo_reduced_flux.assign(n_cells, 0.0);
  }
  if (link_exists(file, "holo/mass_q")) {
    out.state.holo_mass_q = read_vector_dataset_checked<double>(
        file, "holo/mass_q", H5T_NATIVE_DOUBLE, n_cells);
  } else {
    out.state.holo_mass_q.assign(n_cells, 0.0);
  }

// ---- DDMC mode map reconstruction from particles

  std::vector<std::int8_t> ddmc_map_from_particles;
  restore_particles(file, out.photon_pool, &ddmc_map_from_particles, out.state.rho.size(), n_groups);
  if (!has_checkpoint_ddmc_map) {
    if (!ddmc_map_from_particles.empty() && out.photon_pool.n_alive > 0) {
      out.state.ddmc_mode_map = std::move(ddmc_map_from_particles);
      out.state.ddmc_mode_map_valid = true;
      core::log_warning(
          "Restart checkpoint missing radiation/ddmc_flag; reconstructed mode map from particle "
          "states.");
    } else {
      out.state.ddmc_mode_map_valid = false;
      core::log_warning(
          "Restart checkpoint missing radiation/ddmc_flag; defaulting mode map to IMC "
          "(recomputed during transport setup).");
    }
  }
