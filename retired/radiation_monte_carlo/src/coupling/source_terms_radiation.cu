// Retired with the Monte Carlo radiation on 2026-09-29: the radiation source injection of Radiation.mode
// "imc_ddmc" (the matter update from the IMC/DDMC deposition and emission tallies, with the net electron source
// smoothing of Radiation.imc.net_e_source_smoothing and conservative_smoother), cut out of
// src/coupling/source_terms.cu as it was at 5bc8f6ce3. Not built. The helpers it calls in the anonymous namespace of
// that file (make_source_material_params, source_cell_table_selector, compute_effective_A_gamma, the table-EOS
// closure helpers) stayed there.

namespace tenryu::coupling {
namespace {

constexpr double active_W_for_smoothing = 0.5;
constexpr std::int8_t kTransportModeDiffusion = 3;

void compute_raw_net_radiation_source_terms(const std::vector<double>& rad_dep,
                                            const std::vector<double>& rad_emit,
                                            const int n_cells,
                                            const int n_groups,
                                            std::vector<double>& raw_delta_E) {
  raw_delta_E.assign(static_cast<std::size_t>(std::max(n_cells, 0)), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cell_base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
    double dep_sum = 0.0;
    double emit_sum = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = cell_base + static_cast<std::size_t>(g);
      dep_sum += rad_dep[idx];
      emit_sum += rad_emit[idx];
    }
    raw_delta_E[static_cast<std::size_t>(c)] = dep_sum - emit_sum;
  }
}

struct NetESourceSmoothingDiagnostics {
  int faces_total = 0;
  int faces_active = 0;
  double alpha_mean = 0.0;
  double alpha_min = 0.0;
};

NetESourceSmoothingDiagnostics compute_gradient_adaptive_smoothing_diagnostics(
    const core::Config::RadiationConfig::ImcConfig::NetElectronSourceSmoothingConfig&
        smoothing_cfg,
    const std::vector<double>& mass,
    const std::vector<double>& node_r,
    const std::vector<double>& sigma_R_max,
    const std::vector<int>& dominant_material,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& Te,
    const std::vector<double>& rho) {
  constexpr double kMassEps = 1.0e-30;
  constexpr double kLogFloor = 1.0e-30;
  const int n_cells = static_cast<int>(mass.size());
  NetESourceSmoothingDiagnostics diag;
  diag.faces_total = std::max(n_cells - 1, 0);
  for (int left = 0; left + 1 < n_cells; ++left) {
    const int right = left + 1;
    if (cell_is_void[static_cast<std::size_t>(left)] != static_cast<std::uint8_t>(0) ||
        cell_is_void[static_cast<std::size_t>(right)] != static_cast<std::uint8_t>(0)) {
      continue;
    }
    if (dominant_material[static_cast<std::size_t>(left)] !=
        dominant_material[static_cast<std::size_t>(right)]) {
      continue;
    }
    const double mass_left = mass[static_cast<std::size_t>(left)];
    const double mass_right = mass[static_cast<std::size_t>(right)];
    if (!(mass_left > kMassEps) || !(mass_right > kMassEps)) {
      continue;
    }

    const double dx_left =
        std::fmax(node_r[static_cast<std::size_t>(left + 1)] -
                     node_r[static_cast<std::size_t>(left)],
                 0.0);
    const double dx_right =
        std::fmax(node_r[static_cast<std::size_t>(right + 1)] -
                     node_r[static_cast<std::size_t>(right)],
                 0.0);
    const double tau_left =
        std::fmax(sigma_R_max[static_cast<std::size_t>(left)], 0.0) * dx_left;
    const double tau_right =
        std::fmax(sigma_R_max[static_cast<std::size_t>(right)], 0.0) * dx_right;
    if (std::fmin(tau_left, tau_right) < smoothing_cfg.tau_threshold) {
      continue;
    }

    const double dln_Te = std::fabs(std::log(
        std::fmax(Te[static_cast<std::size_t>(right)], kLogFloor) /
        std::fmax(Te[static_cast<std::size_t>(left)], kLogFloor)));
    const double dln_rho = std::fabs(std::log(
        std::fmax(rho[static_cast<std::size_t>(right)], kLogFloor) /
        std::fmax(rho[static_cast<std::size_t>(left)], kLogFloor)));
    const double Te_arg = dln_Te / smoothing_cfg.grad_Te_scale;
    const double rho_arg = dln_rho / smoothing_cfg.grad_rho_scale;
    const double alpha_face =
        smoothing_cfg.alpha * std::exp(-(Te_arg * Te_arg)) *
        std::exp(-(rho_arg * rho_arg));
    diag.alpha_mean += alpha_face;
    if (diag.faces_active == 0 || alpha_face < diag.alpha_min) {
      diag.alpha_min = alpha_face;
    }
    ++diag.faces_active;
  }
  if (diag.faces_active > 0) {
    diag.alpha_mean /= static_cast<double>(diag.faces_active);
  } else {
    diag.alpha_mean = 0.0;
    diag.alpha_min = 0.0;
  }
  return diag;
}

__device__ __forceinline__ double smooth_face_flux_1d_device(
    const double* __restrict__ H_raw,
    const double* __restrict__ mass,
    const double* __restrict__ node_r,
    const double* __restrict__ sigma_R_max,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const int* __restrict__ dominant_material,
    const std::uint8_t* __restrict__ cell_is_void,
    const int left,
    const int right,
    const int n_cells,
    const double alpha,
    const double tau_threshold,
    const double grad_Te_scale,
    const double grad_rho_scale,
    const bool gradient_adaptive) {
  constexpr double kMassEps = 1.0e-30;
  constexpr double kLogFloor = 1.0e-30;
  if (left < 0 || right >= n_cells) {
    return 0.0;
  }
  if (cell_is_void[left] != static_cast<std::uint8_t>(0) ||
      cell_is_void[right] != static_cast<std::uint8_t>(0)) {
    return 0.0;
  }
  if (dominant_material[left] != dominant_material[right]) {
    return 0.0;
  }

  const double mass_left = mass[left];
  const double mass_right = mass[right];
  if (!(mass_left > kMassEps) || !(mass_right > kMassEps)) {
    return 0.0;
  }

  const double dx_left = fmax(node_r[left + 1] - node_r[left], 0.0);
  const double dx_right = fmax(node_r[right + 1] - node_r[right], 0.0);
  const double tau_left = fmax(sigma_R_max[left], 0.0) * dx_left;
  const double tau_right = fmax(sigma_R_max[right], 0.0) * dx_right;
  if (fmin(tau_left, tau_right) < tau_threshold) {
    return 0.0;
  }

  double alpha_face = alpha;
  if (gradient_adaptive) {
    const double dln_Te =
        fabs(log(fmax(Te[right], kLogFloor) / fmax(Te[left], kLogFloor)));
    const double dln_rho =
        fabs(log(fmax(rho[right], kLogFloor) / fmax(rho[left], kLogFloor)));
    const double Te_arg = dln_Te / grad_Te_scale;
    const double rho_arg = dln_rho / grad_rho_scale;
    alpha_face = alpha * exp(-(Te_arg * Te_arg)) * exp(-(rho_arg * rho_arg));
  }

  const double m_face =
      2.0 * mass_left * mass_right / fmax(mass_left + mass_right, kMassEps);
  const double e_left = H_raw[left] / mass_left;
  const double e_right = H_raw[right] / mass_right;
  return alpha_face * m_face * (e_left - e_right);
}

__device__ __forceinline__ double smooth_face_flux_2d_device(
    const double* __restrict__ H_raw,
    const double* __restrict__ mass,
    const double* __restrict__ sigma_R_max,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const int* __restrict__ dominant_material,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ difference_W,
    const int left,
    const int right,
    const int n_cells,
    const double alpha,
    const double tau_threshold,
    const double grad_Te_scale,
    const double grad_rho_scale,
    const bool gradient_adaptive,
    const double tau_left,
    const double tau_right) {
  (void)sigma_R_max;
  constexpr double kDifferenceWBarrier = 0.5;
  constexpr double kMassEps = 1.0e-30;
  constexpr double kLogFloor = 1.0e-30;
  if (left < 0 || right >= n_cells) {
    return 0.0;
  }
  if (cell_is_void[left] != static_cast<std::uint8_t>(0) ||
      cell_is_void[right] != static_cast<std::uint8_t>(0)) {
    return 0.0;
  }
  if (difference_W != nullptr &&
      (!(difference_W[left] < kDifferenceWBarrier) ||
       !(difference_W[right] < kDifferenceWBarrier))) {
    return 0.0;
  }
  if (dominant_material[left] != dominant_material[right]) {
    return 0.0;
  }

  const double mass_left = mass[left];
  const double mass_right = mass[right];
  if (!(mass_left > kMassEps) || !(mass_right > kMassEps)) {
    return 0.0;
  }

  if (fmin(tau_left, tau_right) < tau_threshold) {
    return 0.0;
  }

  double alpha_face = alpha;
  if (gradient_adaptive) {
    const double dln_Te =
        fabs(log(fmax(Te[right], kLogFloor) / fmax(Te[left], kLogFloor)));
    const double dln_rho =
        fabs(log(fmax(rho[right], kLogFloor) / fmax(rho[left], kLogFloor)));
    const double Te_arg = dln_Te / grad_Te_scale;
    const double rho_arg = dln_rho / grad_rho_scale;
    alpha_face = alpha * exp(-(Te_arg * Te_arg)) * exp(-(rho_arg * rho_arg));
  }

  const double m_face =
      2.0 * mass_left * mass_right / fmax(mass_left + mass_right, kMassEps);
  const double e_left = H_raw[left] / mass_left;
  const double e_right = H_raw[right] / mass_right;
  return alpha_face * m_face * (e_left - e_right);
}

__global__ void smooth_net_electron_source_terms_1d_kernel(
    double* __restrict__ H_apply,
    const double* __restrict__ H_raw,
    const double* __restrict__ mass,
    const double* __restrict__ node_r,
    const double* __restrict__ sigma_R_max,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const int* __restrict__ dominant_material,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_cells,
    const double alpha,
    const double tau_threshold,
    const double grad_Te_scale,
    const double grad_rho_scale,
    const bool gradient_adaptive) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double F_left =
      (c > 0) ? smooth_face_flux_1d_device(H_raw, mass, node_r, sigma_R_max,
                                           Te, rho, dominant_material, cell_is_void,
                                           c - 1, c, n_cells, alpha,
                                           tau_threshold, grad_Te_scale,
                                           grad_rho_scale, gradient_adaptive)
              : 0.0;
  const double F_right =
      (c + 1 < n_cells)
          ? smooth_face_flux_1d_device(H_raw, mass, node_r, sigma_R_max,
                                       Te, rho, dominant_material, cell_is_void,
                                       c, c + 1, n_cells, alpha, tau_threshold,
                                       grad_Te_scale, grad_rho_scale,
                                       gradient_adaptive)
          : 0.0;
  H_apply[c] = H_raw[c] + F_left - F_right;
}

__global__ void conservative_smooth_delta_E_1d_kernel(
    double* __restrict__ H_out,
    const double* __restrict__ H_in,
    const double* __restrict__ mass,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_cells,
    const double alpha) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (cell_is_void[c] != 0u) {
    H_out[c] = H_in[c];
    return;
  }

  constexpr double kMassEps = 1.0e-30;
  const double mass_c = fmax(mass[c], kMassEps);
  const double e_c = H_in[c] / mass_c;

  double F_left = 0.0;
  if (c > 0 && cell_is_void[c - 1] == 0u) {
    const double mass_l = fmax(mass[c - 1], kMassEps);
    const double m_face = 2.0 * mass_l * mass_c / (mass_l + mass_c);
    F_left = alpha * m_face * (H_in[c - 1] / mass_l - e_c);
  }
  double F_right = 0.0;
  if (c + 1 < n_cells && cell_is_void[c + 1] == 0u) {
    const double mass_r = fmax(mass[c + 1], kMassEps);
    const double m_face = 2.0 * mass_c * mass_r / (mass_c + mass_r);
    F_right = alpha * m_face * (e_c - H_in[c + 1] / mass_r);
  }
  H_out[c] = H_in[c] + F_left - F_right;
}

__global__ void smooth_net_electron_source_terms_2d_kernel(
    double* __restrict__ H_apply,
    const double* __restrict__ H_raw,
    const double* __restrict__ mass,
    const double* __restrict__ vol,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ sigma_R_max,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const int* __restrict__ dominant_material,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ difference_W,
    const int nr,
    const int nz,
    const double alpha,
    const double tau_threshold,
    const double grad_Te_scale,
    const double grad_rho_scale,
    const bool gradient_adaptive) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const tenryu::mesh::CellWidths2D widths =
      tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz, c);
  const double sigma_this = fmax(sigma_R_max[c], 0.0);
  const double tau_R_this = sigma_this * fmax(widths.h_R, 0.0);
  const double tau_Z_this = sigma_this * fmax(widths.h_Z, 0.0);

  double flux_sum = 0.0;
  if (i > 0) {
    const int neighbor = c - nz;
    const tenryu::mesh::CellWidths2D neighbor_widths =
        tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz,
                                             neighbor);
    const double tau_R_neighbor =
        fmax(sigma_R_max[neighbor], 0.0) * fmax(neighbor_widths.h_R, 0.0);
    flux_sum += smooth_face_flux_2d_device(
        H_raw, mass, sigma_R_max, Te, rho, dominant_material, cell_is_void,
        difference_W, neighbor, c, n_cells, alpha, tau_threshold, grad_Te_scale,
        grad_rho_scale, gradient_adaptive, tau_R_neighbor, tau_R_this);
  }
  if (i + 1 < nr) {
    const int neighbor = c + nz;
    const tenryu::mesh::CellWidths2D neighbor_widths =
        tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz,
                                             neighbor);
    const double tau_R_neighbor =
        fmax(sigma_R_max[neighbor], 0.0) * fmax(neighbor_widths.h_R, 0.0);
    flux_sum -= smooth_face_flux_2d_device(
        H_raw, mass, sigma_R_max, Te, rho, dominant_material, cell_is_void,
        difference_W, c, neighbor, n_cells, alpha, tau_threshold, grad_Te_scale,
        grad_rho_scale, gradient_adaptive, tau_R_this, tau_R_neighbor);
  }
  if (j > 0) {
    const int neighbor = c - 1;
    const tenryu::mesh::CellWidths2D neighbor_widths =
        tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz,
                                             neighbor);
    const double tau_Z_neighbor =
        fmax(sigma_R_max[neighbor], 0.0) * fmax(neighbor_widths.h_Z, 0.0);
    flux_sum += smooth_face_flux_2d_device(
        H_raw, mass, sigma_R_max, Te, rho, dominant_material, cell_is_void,
        difference_W, neighbor, c, n_cells, alpha, tau_threshold, grad_Te_scale,
        grad_rho_scale, gradient_adaptive, tau_Z_neighbor, tau_Z_this);
  }
  if (j + 1 < nz) {
    const int neighbor = c + 1;
    const tenryu::mesh::CellWidths2D neighbor_widths =
        tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz,
                                             neighbor);
    const double tau_Z_neighbor =
        fmax(sigma_R_max[neighbor], 0.0) * fmax(neighbor_widths.h_Z, 0.0);
    flux_sum -= smooth_face_flux_2d_device(
        H_raw, mass, sigma_R_max, Te, rho, dominant_material, cell_is_void,
        difference_W, c, neighbor, n_cells, alpha, tau_threshold, grad_Te_scale,
        grad_rho_scale, gradient_adaptive, tau_Z_this, tau_Z_neighbor);
  }

  H_apply[c] = H_raw[c] + flux_sum;
}

__global__ void inject_radiation_source_terms_kernel(
    double* __restrict__ ee,
    double* __restrict__ Te,
    double* __restrict__ Pe,
    double* __restrict__ ei,
    double* __restrict__ Ti,
    double* __restrict__ Pi,
    double* __restrict__ delta_E_rad_prev,
    const double* __restrict__ rho,
    const double* __restrict__ mass,
    const double* __restrict__ zbar,
    const double* __restrict__ cv_e,
    const double* __restrict__ applied_delta_E,
    const double* __restrict__ A_eff,
    const double* __restrict__ gamma_eff,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::uint8_t* __restrict__ diffusion_cell,
    const std::uint8_t* __restrict__ holo_source_cell,
    const int n_cells,
    const double te_floor,
    const double ti_floor,
    const bool use_two_temp,
    const bool has_table_eos,
    const bool use_first_cv_override,
    const double cv_e_override,
    const double eos_T_ref_eV,
    const tenryu::materials::DeviceEOSTableView tab_ion_first,
    const tenryu::materials::DeviceEOSTableView tab_ele_first,
    const tenryu::materials::DeviceEOSTableView tab_total_first,
    const tenryu::materials::CellEOSTableSelector cell_tables,
    double* __restrict__ floor_energy,
    double* __restrict__ skipped_energy,
    int* __restrict__ clamp_count,
    const int energy_authoritative) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double delta_E = applied_delta_E[c];
  delta_E_rad_prev[c] = delta_E;
  if (holo_source_cell != nullptr &&
      holo_source_cell[c] != static_cast<std::uint8_t>(0)) {
    return;
  }
  if (diffusion_cell[c] != static_cast<std::uint8_t>(0)) {
    delta_E_rad_prev[c] = 0.0;
    return;
  }
  if (cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    skipped_energy[c] += delta_E;
    return;
  }

  const double rho_c = rho[c];
  const double mass_c = mass[c];
  if (mass_c < 1.0e-30) {
    skipped_energy[c] += delta_E;
    return;
  }

  // Per-cell dominant-material tables (multi-material closure, 2026-09-14);
  // a 1D cell whose material has no table closes with the ideal gas
  // (source_cell_table_selector).
  const tenryu::materials::DeviceEOSTableView tab_ion = cell_tables.ion(c, tab_ion_first);
  const tenryu::materials::DeviceEOSTableView tab_ele =
      cell_tables.electron(c, tab_ele_first);
  // 1T: the total energy closes on the cell's total table (2026-09-23; the 1T
  // injection used the ideal-gas branch for every cell).
  const tenryu::materials::DeviceEOSTableView tab_total = cell_tables.total(c, tab_total_first);
  const tenryu::materials::DeviceEOSTableView tab_closure = use_two_temp ? tab_ele : tab_total;

  // Heat-capacity override of the cell's material when the materials'
  // closure parameters differ (per-cell 1D runs), else the run-level one
  // (2026-09-24).
  const materials::MaterialClosureParams* cp_c = cell_tables.closure_of(c);
  const bool use_cv_override_c =
      (cp_c != nullptr) ? (cp_c->cv_e_override > 0.0) : use_first_cv_override;
  const double cv_override_c = (cp_c != nullptr) ? cp_c->cv_e_override : cv_e_override;
  const double T_ref_c = (cp_c != nullptr) ? cp_c->eos_T_ref_eV : eos_T_ref_eV;
  const double A_c = A_eff[c];
  const double gm1_c = gamma_eff[c] - 1.0;
  const double cv_mass_i =
      fmax(tenryu::core::constants::eV_to_erg /
               (A_c * tenryu::core::constants::proton_mass * gm1_c),
           1.0e-30);
  if (!use_two_temp) {
    ee[c] += ei[c];
  }
  ee[c] += delta_E / mass_c;
  const double ee_before_floor = ee[c];

  const double rho_safe = fmax(rho_c, 1.0e-30);
  const bool use_table_eos_closure = has_table_eos && tab_closure.n_rho > 0;
  double Te_raw = 0.0;
  double Te_new = te_floor;
  if (use_table_eos_closure) {
    const auto rb_e = tenryu::materials::find_rho_bracket(tab_closure, rho_safe);
    Te_raw = tenryu::materials::device_eos_T_from_e_monotone(
        tab_closure, rb_e, ee[c]);
    if (!isfinite(Te_raw) || Te_raw < te_floor) {
      Te_new = te_floor;
      // Energy-authoritative: keep the sub-floor energy (only the temperature
      // is floored); the table value is written only for non-finite input.
      if (energy_authoritative == 0 || !isfinite(Te_raw)) {
        const double logTe = log(fmax(Te_new, 1.0e-300));
        ee[c] = tenryu::materials::device_eos_energy(tab_closure, rb_e, logTe);
      }
    } else {
      Te_new = Te_raw;
    }
    const double logTe = log(fmax(Te_new, 1.0e-300));
    Pe[c] = tenryu::materials::device_eos_pressure(tab_closure, rb_e, logTe);
  } else if (use_cv_override_c && T_ref_c > 0.0) {
    const double T_ref3 = T_ref_c * T_ref_c * T_ref_c;
    const double alpha0 = cv_override_c / (4.0 * T_ref3);
    const double arg = ee[c] * rho_safe / alpha0;
    Te_raw = (arg > 0.0) ? pow(arg, 0.25) : 0.0;
    if (!isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = fmax(Te_raw, te_floor);
    const double Te2 = Te_new * Te_new;
    ee[c] = alpha0 * Te2 * Te2 / rho_safe;
  } else {
    const double z = fmax(zbar[c], 0.0);
    double cv_mass_e = 0.0;
    bool cv_mass_e_is_total = false;
    if (use_cv_override_c) {
      cv_mass_e = cv_override_c / rho_safe;
    } else if (cv_e != nullptr && cv_e[c] > 0.0) {
      cv_mass_e = cv_e[c];
      // In 1T the state heat capacity is the total one (the 1T hydro closure
      // stores it there); the ion part used to be added a second time.
      cv_mass_e_is_total = !use_two_temp;
    } else {
      cv_mass_e = z * tenryu::core::constants::eV_to_erg /
                  (A_c * tenryu::core::constants::proton_mass * gm1_c);
    }
    cv_mass_e = fmax(cv_mass_e, 1.0e-30);
    const double cv_mass_total =
        use_two_temp ? cv_mass_e
                     : ((use_cv_override_c || cv_mass_e_is_total)
                            ? cv_mass_e
                            : fmax(cv_mass_e + cv_mass_i, 1.0e-30));
    Te_raw = ee[c] / cv_mass_total;
    if (!isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = fmax(Te_raw, te_floor);
    ee[c] = cv_mass_total * Te_new;
  }

  if (Te_new > Te_raw) {
    atomicAdd(clamp_count, 1);
  }
  const double de_floor_e = ee[c] - ee_before_floor;
  if (de_floor_e > 0.0) {
    floor_energy[c] += mass_c * de_floor_e;
  }

  Te[c] = Te_new;
  if (!use_table_eos_closure) {
    Pe[c] = gm1_c * rho_c * ee[c];
  }
  if (use_two_temp) {
    if (has_table_eos && tab_ion.n_rho > 0) {
      const auto rb_i = tenryu::materials::find_rho_bracket(tab_ion, rho_safe);
      const double Ti_raw_table =
          tenryu::materials::device_eos_T_from_e_monotone(tab_ion, rb_i, ei[c]);
      const double ei_before_floor = ei[c];
      if (!isfinite(Ti_raw_table) || Ti_raw_table < ti_floor) {
        atomicAdd(clamp_count, 1);
        Ti[c] = ti_floor;
        if (energy_authoritative == 0 || !isfinite(Ti_raw_table)) {
          const double logTi = log(fmax(ti_floor, 1.0e-300));
          ei[c] = tenryu::materials::device_eos_energy(tab_ion, rb_i, logTi);
        }
      } else {
        Ti[c] = Ti_raw_table;
      }
      const double logTi = log(fmax(Ti[c], 1.0e-300));
      Pi[c] = tenryu::materials::device_eos_pressure(tab_ion, rb_i, logTi);
      const double de_floor_i = ei[c] - ei_before_floor;
      if (de_floor_i > 0.0) {
        floor_energy[c] += mass_c * de_floor_i;
      }
    } else {
      const double Ti_prev = isfinite(Ti[c]) ? Ti[c] : 0.0;
      if (Ti_prev < ti_floor) {
        atomicAdd(clamp_count, 1);
        const double ei_before_floor = ei[c];
        Ti[c] = ti_floor;
        ei[c] = cv_mass_i * ti_floor;
        Pi[c] = gm1_c * rho_c * ei[c];
        const double de_floor_i = ei[c] - ei_before_floor;
        if (de_floor_i > 0.0) {
          floor_energy[c] += mass_c * de_floor_i;
        }
      }
    }
  } else {
    Ti[c] = Te_new;
    ei[c] = 0.0;
    Pi[c] = 0.0;
  }
}

void smooth_net_electron_source_terms_1d(
    const core::Config::RadiationConfig::ImcConfig::NetElectronSourceSmoothingConfig&
        smoothing_cfg,
    const core::State& state,
    const std::vector<double>& sigma_R_max,
    const std::vector<int>& dominant_material,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& raw_delta_E,
    std::vector<double>& applied_delta_E) {
  const int n_cells = static_cast<int>(raw_delta_E.size());
  applied_delta_E = raw_delta_E;
  TENRYU_ASSERT(smoothing_cfg.passes >= 0,
                "smooth_net_electron_source_terms_1d passes must be >= 0");
  const int smooth_passes = smoothing_cfg.passes;
  if (n_cells < 2 || !(smoothing_cfg.alpha > 0.0) || smooth_passes == 0) {
    return;
  }

  TENRYU_ASSERT(static_cast<int>(sigma_R_max.size()) == n_cells,
                "smooth_net_electron_source_terms_1d sigma_R_max size mismatch");
  TENRYU_ASSERT(static_cast<int>(dominant_material.size()) == n_cells,
                "smooth_net_electron_source_terms_1d dominant_material size mismatch");
  TENRYU_ASSERT(static_cast<int>(cell_is_void.size()) == n_cells,
                "smooth_net_electron_source_terms_1d cell_is_void size mismatch");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_cells + 1),
                "smooth_net_electron_source_terms_1d x_r size mismatch");
  TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_1d mass size mismatch");
  TENRYU_ASSERT(state.Te.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_1d Te size mismatch");
  TENRYU_ASSERT(state.rho.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_1d rho size mismatch");
  if (smoothing_cfg.gradient_adaptive) {
    TENRYU_ASSERT(smoothing_cfg.grad_Te_scale > 0.0,
                  "smooth_net_electron_source_terms_1d grad_Te_scale must be > 0");
    TENRYU_ASSERT(smoothing_cfg.grad_rho_scale > 0.0,
                  "smooth_net_electron_source_terms_1d grad_rho_scale must be > 0");
  }

  NetESourceSmoothingDiagnostics smoothing_diag;
  if (smoothing_cfg.gradient_adaptive) {
    std::vector<double> host_mass(static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> host_node_r(static_cast<std::size_t>(n_cells + 1), 0.0);
    std::vector<double> host_Te(static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> host_rho(static_cast<std::size_t>(n_cells), 0.0);
    state.mass.copy_to_host(host_mass.data());
    state.x_r.copy_to_host(host_node_r.data());
    state.Te.copy_to_host(host_Te.data());
    state.rho.copy_to_host(host_rho.data());
    smoothing_diag = compute_gradient_adaptive_smoothing_diagnostics(
        smoothing_cfg, host_mass, host_node_r, sigma_R_max, dominant_material,
        cell_is_void, host_Te, host_rho);
  }

  double* d_H_raw = nullptr;
  double* d_H_apply = nullptr;
  double* d_sigma_R_max = nullptr;
  int* d_dominant_material = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;

  d_H_raw = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_H_raw",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_H_apply = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_H_apply",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_sigma_R_max = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_sigma_R_max",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_dominant_material = static_cast<int*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_dominant_material",
      sizeof(int) * static_cast<std::size_t>(n_cells)));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_cell_is_void",
      sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells)));

  cuda_check(cudaMemcpy(d_H_raw, raw_delta_E.data(),
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_1d copy H_raw failed");
  cuda_check(cudaMemcpy(d_sigma_R_max, sigma_R_max.data(),
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_1d copy sigma_R_max failed");
  cuda_check(cudaMemcpy(d_dominant_material, dominant_material.data(),
                        sizeof(int) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_1d copy dominant_material failed");
  cuda_check(cudaMemcpy(d_cell_is_void, cell_is_void.data(),
                        sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_1d copy cell_is_void failed");

  const int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  for (int pass = 0; pass < smooth_passes; ++pass) {
    const double* const d_H_in = (pass % 2 == 0) ? d_H_raw : d_H_apply;
    double* const d_H_out = (pass % 2 == 0) ? d_H_apply : d_H_raw;
    smooth_net_electron_source_terms_1d_kernel<<<blocks, threads>>>(
        d_H_out, d_H_in, state.mass.data(), state.x_r.data(), d_sigma_R_max,
        state.Te.data(), state.rho.data(), d_dominant_material, d_cell_is_void,
        n_cells, smoothing_cfg.alpha, smoothing_cfg.tau_threshold,
        smoothing_cfg.grad_Te_scale, smoothing_cfg.grad_rho_scale,
        smoothing_cfg.gradient_adaptive);
    cuda_check(cudaGetLastError(),
               "smooth_net_electron_source_terms_1d kernel launch failed");
  }

  const double* const d_H_result =
      (smooth_passes % 2 == 0) ? d_H_raw : d_H_apply;
  cuda_check(cudaMemcpy(applied_delta_E.data(), d_H_result,
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyDeviceToHost),
             "smooth_net_electron_source_terms_1d copy H_apply failed");

  if (smoothing_cfg.gradient_adaptive) {
    std::ostringstream oss;
    oss << "[net_e_smooth] step=" << (state.step + 1)
        << " faces_total=" << smoothing_diag.faces_total
        << " faces_active=" << smoothing_diag.faces_active
        << " alpha_mean=" << smoothing_diag.alpha_mean
        << " alpha_min=" << smoothing_diag.alpha_min;
    core::log_info(oss.str());
  }
}

void smooth_net_electron_source_terms_2d(
    const core::Config::RadiationConfig::ImcConfig::NetElectronSourceSmoothingConfig&
        smoothing_cfg,
    const core::State& state,
    const std::vector<double>& sigma_R_max,
    const std::vector<int>& dominant_material,
    const std::vector<std::uint8_t>& cell_is_void,
    const double* difference_W,
    const std::vector<double>& raw_delta_E,
    std::vector<double>& applied_delta_E) {
  const int n_cells = static_cast<int>(raw_delta_E.size());
  applied_delta_E = raw_delta_E;
  TENRYU_ASSERT(smoothing_cfg.passes >= 0,
                "smooth_net_electron_source_terms_2d passes must be >= 0");
  const int smooth_passes = smoothing_cfg.passes;
  if (n_cells < 2 || !(smoothing_cfg.alpha > 0.0) || smooth_passes == 0) {
    return;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(nr > 0 && nz > 0,
                "smooth_net_electron_source_terms_2d requires positive topology");
  TENRYU_ASSERT(n_cells == nr * nz,
                "smooth_net_electron_source_terms_2d topology size mismatch");
  TENRYU_ASSERT(static_cast<int>(sigma_R_max.size()) == n_cells,
                "smooth_net_electron_source_terms_2d sigma_R_max size mismatch");
  TENRYU_ASSERT(static_cast<int>(dominant_material.size()) == n_cells,
                "smooth_net_electron_source_terms_2d dominant_material size mismatch");
  TENRYU_ASSERT(static_cast<int>(cell_is_void.size()) == n_cells,
                "smooth_net_electron_source_terms_2d cell_is_void size mismatch");
  const std::size_t n_nodes =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  TENRYU_ASSERT(state.x_r.size() == n_nodes,
                "smooth_net_electron_source_terms_2d x_r size mismatch");
  TENRYU_ASSERT(state.x_z.size() == n_nodes,
                "smooth_net_electron_source_terms_2d x_z size mismatch");
  TENRYU_ASSERT(state.vol.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_2d vol size mismatch");
  TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_2d mass size mismatch");
  TENRYU_ASSERT(state.Te.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_2d Te size mismatch");
  TENRYU_ASSERT(state.rho.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_2d rho size mismatch");
  if (smoothing_cfg.gradient_adaptive) {
    TENRYU_ASSERT(smoothing_cfg.grad_Te_scale > 0.0,
                  "smooth_net_electron_source_terms_2d grad_Te_scale must be > 0");
    TENRYU_ASSERT(smoothing_cfg.grad_rho_scale > 0.0,
                  "smooth_net_electron_source_terms_2d grad_rho_scale must be > 0");
  }

  double* d_H_raw = nullptr;
  double* d_H_apply = nullptr;
  double* d_sigma_R_max = nullptr;
  int* d_dominant_material = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;

  d_H_raw = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_H_raw",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_H_apply = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_H_apply",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_sigma_R_max = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_sigma_R_max",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_dominant_material = static_cast<int*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_dominant_material",
      sizeof(int) * static_cast<std::size_t>(n_cells)));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_cell_is_void",
      sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells)));

  cuda_check(cudaMemcpy(d_H_raw, raw_delta_E.data(),
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_2d copy H_raw failed");
  cuda_check(cudaMemcpy(d_sigma_R_max, sigma_R_max.data(),
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_2d copy sigma_R_max failed");
  cuda_check(cudaMemcpy(d_dominant_material, dominant_material.data(),
                        sizeof(int) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_2d copy dominant_material failed");
  cuda_check(cudaMemcpy(d_cell_is_void, cell_is_void.data(),
                        sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_2d copy cell_is_void failed");

  const int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  for (int pass = 0; pass < smooth_passes; ++pass) {
    const double* const d_H_in = (pass % 2 == 0) ? d_H_raw : d_H_apply;
    double* const d_H_out = (pass % 2 == 0) ? d_H_apply : d_H_raw;
    smooth_net_electron_source_terms_2d_kernel<<<blocks, threads>>>(
        d_H_out, d_H_in, state.mass.data(), state.vol.data(), state.x_r.data(),
        state.x_z.data(), d_sigma_R_max, state.Te.data(), state.rho.data(),
        d_dominant_material, d_cell_is_void, difference_W, nr, nz,
        smoothing_cfg.alpha, smoothing_cfg.tau_threshold,
        smoothing_cfg.grad_Te_scale, smoothing_cfg.grad_rho_scale,
        smoothing_cfg.gradient_adaptive);
    cuda_check(cudaGetLastError(),
               "smooth_net_electron_source_terms_2d kernel launch failed");
  }

  const double* const d_H_result =
      (smooth_passes % 2 == 0) ? d_H_raw : d_H_apply;
  cuda_check(cudaMemcpy(applied_delta_E.data(), d_H_result,
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyDeviceToHost),
             "smooth_net_electron_source_terms_2d copy H_apply failed");

}

void conservative_smooth_delta_E_1d(
    const core::Config::RadiationConfig::ImcConfig::ConservativeSmootherConfig&
        smoothing_cfg,
    const core::State& state,
    const std::vector<std::uint8_t>& cell_is_void,
    std::vector<double>& applied_delta_E) {
  const int n_cells = static_cast<int>(applied_delta_E.size());
  TENRYU_ASSERT(smoothing_cfg.passes >= 0,
                "conservative_smooth_delta_E_1d passes must be >= 0");
  const int smooth_passes = smoothing_cfg.passes;
  if (n_cells < 2 || !(smoothing_cfg.alpha > 0.0) || smooth_passes == 0) {
    return;
  }

  TENRYU_ASSERT(static_cast<int>(cell_is_void.size()) == n_cells,
                "conservative_smooth_delta_E_1d cell_is_void size mismatch");
  TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells),
                "conservative_smooth_delta_E_1d mass size mismatch");

  double* d_H_a = nullptr;
  double* d_H_b = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;

  const std::size_t cell_bytes =
      sizeof(double) * static_cast<std::size_t>(n_cells);
  const std::size_t mask_bytes =
      sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
  d_H_a = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:conservative_smooth_delta_E_1d:d_H_a", cell_bytes));
  d_H_b = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:conservative_smooth_delta_E_1d:d_H_b", cell_bytes));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:conservative_smooth_delta_E_1d:d_cell_is_void",
      mask_bytes));

  cuda_check(cudaMemcpy(d_H_a, applied_delta_E.data(), cell_bytes,
                        cudaMemcpyHostToDevice),
             "conservative_smooth_delta_E_1d copy H_a failed");
  cuda_check(cudaMemcpy(d_cell_is_void, cell_is_void.data(), mask_bytes,
                        cudaMemcpyHostToDevice),
             "conservative_smooth_delta_E_1d copy cell_is_void failed");

  const int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  for (int pass = 0; pass < smooth_passes; ++pass) {
    const double* const d_H_in = (pass % 2 == 0) ? d_H_a : d_H_b;
    double* const d_H_out = (pass % 2 == 0) ? d_H_b : d_H_a;
    conservative_smooth_delta_E_1d_kernel<<<blocks, threads>>>(
        d_H_out, d_H_in, state.mass.data(), d_cell_is_void, n_cells,
        smoothing_cfg.alpha);
    cuda_check(cudaGetLastError(),
               "conservative_smooth_delta_E_1d kernel launch failed");
  }

  const double* const d_H_result =
      (smooth_passes % 2 == 0) ? d_H_a : d_H_b;
  cuda_check(cudaMemcpy(applied_delta_E.data(), d_H_result, cell_bytes,
                        cudaMemcpyDeviceToHost),
             "conservative_smooth_delta_E_1d copy H_result failed");

}

}  // namespace

double inject_radiation_source_terms_impl(core::State& state,
                                          const core::Config& cfg,
                                          const double dt,
                                          double* E_floor_injected,
                                          int* clamp_count,
                                          const std::vector<double>* sigma_R_max,
                                          const hydro::HydroEOSContext* eos_ctx) {
  const bool verbose_subphase_timing = (cfg.main.verbosity == "verbose");
  using Clock = std::chrono::steady_clock;
  const auto t_inject_start =
      verbose_subphase_timing ? Clock::now() : Clock::time_point{};
  auto t_inject_subphase = t_inject_start;
  double setup_ms = 0.0;
  double diff_cell_ms = 0.0;
  double blocked_ms = 0.0;
  double d2h_ms = 0.0;
  double raw_net_ms = 0.0;
  double smooth_ms = 0.0;
  double gpu_ms = 0.0;
  double reduction_ms = 0.0;
  double finalize_ms = 0.0;
  const auto mark_subphase = [&](double& value) {
    if (verbose_subphase_timing) {
      const auto t_now = Clock::now();
      value += std::chrono::duration<double, std::milli>(
                   t_now - t_inject_subphase)
                   .count();
      t_inject_subphase = t_now;
    }
  };
  const auto log_subphase = [&]() {
    if (verbose_subphase_timing) {
      const double total_ms = setup_ms + diff_cell_ms + blocked_ms + d2h_ms +
                              raw_net_ms + smooth_ms + gpu_ms +
                              reduction_ms + finalize_ms;
      std::ostringstream oss;
      oss << "[inject_subphase] step=" << state.step
          << std::fixed << std::setprecision(2)
          << " setup=" << setup_ms
          << " diff_cell=" << diff_cell_ms
          << " blocked=" << blocked_ms
          << " d2h=" << d2h_ms
          << " raw_net=" << raw_net_ms
          << " smooth=" << smooth_ms
          << " gpu=" << gpu_ms
          << " reduction=" << reduction_ms
          << " finalize=" << finalize_ms
          << " total=" << total_ms << " ms";
      core::log_info(oss.str());
    }
  };

  if (state.rad_dep.empty() || state.rho.empty() || dt <= 0.0) {
    mark_subphase(setup_ms);
    log_subphase();
    return 0.0;
  }
  if (cfg.materials.materials.empty()) {
    mark_subphase(setup_ms);
    log_subphase();
    return 0.0;
  }
  assert_common_source_state_sizes(state, "inject_radiation_source_terms");
  TENRYU_ASSERT(state.rad_emit.empty() || state.rad_emit.size() == state.rad_dep.size(),
                "inject_radiation_source_terms requires rad_emit size == rad_dep size when present");

  const auto& materials = cfg.materials.materials;
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "inject_radiation_source_terms requires at least one non-void material");
  const auto& mat0 = materials[static_cast<std::size_t>(first_nonvoid)];
  const bool has_table_eos = source_table_eos_enabled(cfg, mat0);

  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<double> A_eff;
  std::vector<double> gamma_eff;
  std::vector<int> dominant_material;
  compute_effective_A_gamma(cfg, state, n_cells, A_eff, gamma_eff,
                            &dominant_material);
  TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_radiation_source_terms A_eff size mismatch");
  TENRYU_ASSERT(gamma_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_radiation_source_terms gamma_eff size mismatch");
  TENRYU_ASSERT(dominant_material.size() == static_cast<std::size_t>(n_cells),
                "inject_radiation_source_terms dominant_material size mismatch");

  bool any_cv_e_override = false;
  for (const auto& mat : materials) {
    if (mat.cv_e_override > 0.0) {
      any_cv_e_override = true;
      break;
    }
  }
  const bool use_first_cv_override = any_cv_e_override && mat0.cv_e_override > 0.0;
  mark_subphase(setup_ms);

  TENRYU_ASSERT(state.rad_dep.size() % state.rho.size() == 0,
                "inject_radiation_source_terms requires rad_dep size divisible by rho size");
  const int n_groups =
      (n_cells > 0) ? static_cast<int>(state.rad_dep.size() / state.rho.size()) : 1;
  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  std::vector<std::uint8_t> diffusion_cell(static_cast<std::size_t>(n_cells), 0U);
  if (state.ddmc_mode_map_valid) {
    TENRYU_ASSERT(state.ddmc_mode_map.size() == n_cell_groups,
                  "inject_radiation_source_terms ddmc_mode_map size mismatch");
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t base =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
      for (int g = 0; g < n_groups; ++g) {
        if (state.ddmc_mode_map[base + static_cast<std::size_t>(g)] ==
            kTransportModeDiffusion) {
          diffusion_cell[static_cast<std::size_t>(c)] = 1U;
          break;
        }
      }
    }
  }
  mark_subphase(diff_cell_ms);

  std::vector<std::uint8_t> source_apply_blocked = state.cell_is_void;
  for (int c = 0; c < n_cells; ++c) {
    if (diffusion_cell[static_cast<std::size_t>(c)] != 0U) {
      source_apply_blocked[static_cast<std::size_t>(c)] = 1U;
    }
  }
  const bool smoothing_requested = cfg.radiation.imc.net_e_source_smoothing.enabled;
  if (smoothing_requested && cfg.radiation.imc.difference.enabled &&
      state.difference_W.size() == static_cast<std::size_t>(n_cells)) {
    std::vector<double> difference_W(static_cast<std::size_t>(n_cells), 0.0);
    state.difference_W.copy_to_host(difference_W.data());
    for (int c = 0; c < n_cells; ++c) {
      const double W = difference_W[static_cast<std::size_t>(c)];
      if (!std::isfinite(W) || W >= active_W_for_smoothing) {
        source_apply_blocked[static_cast<std::size_t>(c)] = 1U;
      }
    }
  } else if (smoothing_requested && cfg.radiation.imc.difference.enabled) {
    std::fill(source_apply_blocked.begin(), source_apply_blocked.end(), 1U);
  }

  std::vector<double> rad_dep(state.rad_dep.size(), 0.0);
  std::vector<double> rad_emit(state.rad_dep.size(), 0.0);
  std::vector<double> holo_rad_dep;
  std::vector<double> holo_rad_emit;
  std::vector<double> raw_delta_E;
  std::vector<double> applied_delta_E;

  bool holo_core_present = false;
  bool holo_patch_present = false;
  bool holo_source_owner = false;
  if (cfg.radiation.holo.enabled && state.holo_core_mask_valid) {
    TENRYU_ASSERT(state.holo_core_mask.size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms requires holo_core_mask/rho size match");
    TENRYU_ASSERT(state.holo_patch_mask.size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms requires holo_patch_mask/rho size match");
    TENRYU_ASSERT(state.holo_lo_weight.size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms requires holo_lo_weight/rho size match");
    holo_core_present =
        std::any_of(state.holo_core_mask.begin(),
                    state.holo_core_mask.end(),
                    [](const std::uint8_t value) { return value != 0U; });
    holo_patch_present =
        std::any_of(state.holo_patch_mask.begin(),
                    state.holo_patch_mask.end(),
                    [](const std::uint8_t value) { return value != 0U; });
    if (holo_core_present || holo_patch_present) {
      TENRYU_ASSERT(state.holo_rad_dep.size() == state.rad_dep.size(),
                    "inject_radiation_source_terms requires holo_rad_dep size == rad_dep size");
      TENRYU_ASSERT(state.holo_rad_emit.size() == state.rad_dep.size(),
                    "inject_radiation_source_terms requires holo_rad_emit size == rad_dep size");
    }
  }

  const bool smoothing_supported =
      smoothing_requested && (state.mesh.dim == 1 || state.mesh.dim == 2);
  if (smoothing_supported) {
    TENRYU_ASSERT(sigma_R_max != nullptr,
                  "inject_radiation_source_terms requires sigma_R_max when "
                  "Radiation.imc.net_e_source_smoothing.enabled");
    TENRYU_ASSERT(sigma_R_max->size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms sigma_R_max size mismatch");
  }
  mark_subphase(blocked_ms);

  state.rad_dep.copy_to_host(rad_dep.data());
  if (state.rad_emit.size() == state.rad_dep.size() && !state.rad_emit.empty()) {
    state.rad_emit.copy_to_host(rad_emit.data());
  }
  if (holo_core_present || holo_patch_present) {
    holo_rad_dep.assign(state.rad_dep.size(), 0.0);
    holo_rad_emit.assign(state.rad_dep.size(), 0.0);
    state.holo_rad_dep.copy_to_host(holo_rad_dep.data());
    state.holo_rad_emit.copy_to_host(holo_rad_emit.data());
    holo_source_owner = state.holo_lo_source_valid;
    if (!holo_source_owner) {
      for (int c = 0; c < n_cells && !holo_source_owner; ++c) {
        const std::size_t c_idx = static_cast<std::size_t>(c);
        if (state.holo_core_mask[c_idx] == 0U &&
            state.holo_patch_mask[c_idx] == 0U) {
          continue;
        }
        const std::size_t cell_base =
            c_idx * static_cast<std::size_t>(n_groups);
        for (int g = 0; g < n_groups; ++g) {
          const std::size_t idx = cell_base + static_cast<std::size_t>(g);
          if (holo_rad_dep[idx] != 0.0 || holo_rad_emit[idx] != 0.0) {
            holo_source_owner = true;
            break;
          }
        }
      }
    }
  }
  const bool has_table_cv_e = !state.cv_e.empty();

  const double te_floor = cfg.numerics.floors.Te;
  const double ti_floor = cfg.numerics.floors.Ti;
  const bool use_two_temp = cfg.main.two_temperature;
  const bool sn_material_coupling = cfg.radiation.holo.sn_material_coupling;
  const bool sn_qd_lo_updates_material_directly =
      sn_material_coupling && state.mesh.dim == 1 &&
      cfg.radiation.holo.solver == "quasidiffusion_1d";
  const bool holo_updates_material_directly =
      holo_source_owner && (!sn_material_coupling ||
                            sn_qd_lo_updates_material_directly);
  mark_subphase(d2h_ms);

  compute_raw_net_radiation_source_terms(rad_dep, rad_emit, n_cells, n_groups,
                                         raw_delta_E);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_idx = static_cast<std::size_t>(c);
    const bool holo_core_cell =
        holo_source_owner && state.holo_core_mask[c_idx] != 0U;
    const bool holo_patch_cell =
        holo_source_owner && state.holo_patch_mask[c_idx] != 0U;
    if (holo_core_cell || holo_patch_cell) {
      if (diffusion_cell[c_idx] != 0U) {
        TENRYU_ASSERT(false,
                      "HOLO source ownership cannot overlap deterministic diffusion cells");
      }
      double holo_delta_E = 0.0;
      const std::size_t cell_base =
          c_idx * static_cast<std::size_t>(n_groups);
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t idx = cell_base + static_cast<std::size_t>(g);
        holo_delta_E += holo_rad_dep[idx] - holo_rad_emit[idx];
      }
      if (holo_core_cell) {
        raw_delta_E[c_idx] = holo_delta_E;
      } else {
        const double w_raw = state.holo_lo_weight[c_idx];
        const double w = std::isfinite(w_raw) ? std::clamp(w_raw, 0.0, 1.0) : 0.0;
        raw_delta_E[c_idx] =
            w * holo_delta_E + (1.0 - w) * raw_delta_E[c_idx];
      }
      source_apply_blocked[c_idx] = 1U;
      continue;
    }
    if (diffusion_cell[static_cast<std::size_t>(c)] != 0U) {
      raw_delta_E[static_cast<std::size_t>(c)] = 0.0;
    }
  }
  applied_delta_E = raw_delta_E;
  mark_subphase(raw_net_ms);

  if (smoothing_supported) {
    if (state.mesh.dim == 1) {
      smooth_net_electron_source_terms_1d(
          cfg.radiation.imc.net_e_source_smoothing, state, *sigma_R_max,
          dominant_material, source_apply_blocked, raw_delta_E, applied_delta_E);
    } else if (state.mesh.dim == 2) {
      const double* const difference_W =
          (cfg.radiation.imc.difference.enabled &&
           state.difference_W.size() == static_cast<std::size_t>(n_cells))
              ? state.difference_W.data()
              : nullptr;
      smooth_net_electron_source_terms_2d(
          cfg.radiation.imc.net_e_source_smoothing, state, *sigma_R_max,
          dominant_material, source_apply_blocked, difference_W, raw_delta_E,
          applied_delta_E);
    }
  }
  const auto& cons_smooth = cfg.radiation.imc.conservative_smoother;
  if (cons_smooth.enabled && cons_smooth.passes > 0 &&
      cons_smooth.alpha > 0.0 && n_cells >= 2 && state.mesh.dim == 1) {
    std::vector<std::uint8_t> conservative_source_blocked = state.cell_is_void;
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_idx = static_cast<std::size_t>(c);
      if (diffusion_cell[c_idx] != 0U ||
          (holo_source_owner &&
           (state.holo_core_mask[c_idx] != 0U ||
            state.holo_patch_mask[c_idx] != 0U))) {
        conservative_source_blocked[c_idx] = 1U;
      }
    }
    conservative_smooth_delta_E_1d(cons_smooth, state,
                                   conservative_source_blocked, applied_delta_E);
  }
  mark_subphase(smooth_ms);

  SourceEOSTableViews table_views;
  if (has_table_eos) {
    const int ref = source_table_reference_material(cfg);
    table_views = select_source_eos_table_views(
        materials[static_cast<std::size_t>(ref)], ref, eos_ctx);
    TENRYU_ASSERT(!use_two_temp ||
                      (table_views.electron.n_rho > 0 && table_views.ion.n_rho > 0),
                  "inject_radiation_source_terms requires non-empty device EOS tables");
  }
  if (has_table_cv_e) {
    TENRYU_ASSERT(state.cv_e.size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms cv_e size mismatch");
  }
  if (state.delta_E_rad_prev.size() != static_cast<std::size_t>(n_cells)) {
    state.delta_E_rad_prev.reset(static_cast<std::size_t>(n_cells));
  }

  double* d_applied_delta_E = nullptr;
  double* d_A_eff = nullptr;
  double* d_gamma_eff = nullptr;
  double* d_reduction = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;
  std::uint8_t* d_diffusion_cell = nullptr;
  std::uint8_t* d_holo_source_cell = nullptr;
  int* d_clamp_count = nullptr;

  const std::size_t cell_bytes =
      sizeof(double) * static_cast<std::size_t>(n_cells);
  const std::size_t mask_bytes =
      sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
  d_applied_delta_E = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_applied_delta_E",
      cell_bytes));
  d_A_eff = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_A_eff", cell_bytes));
  d_gamma_eff = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_gamma_eff",
      cell_bytes));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_cell_is_void",
      mask_bytes));
  d_diffusion_cell = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_diffusion_cell",
      mask_bytes));
  if (holo_updates_material_directly) {
    d_holo_source_cell = static_cast<std::uint8_t*>(core::device_scratch_acquire(
        "source_terms:inject_radiation_source_terms_impl:d_holo_source_cell",
        mask_bytes));
  }
  d_reduction = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_reduction",
      2 * sizeof(double)));
  d_clamp_count = static_cast<int*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_clamp_count",
      sizeof(int)));

  cuda_check(cudaMemcpy(d_applied_delta_E, applied_delta_E.data(), cell_bytes,
                        cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy applied_delta_E failed");
  cuda_check(cudaMemcpy(d_A_eff, A_eff.data(), cell_bytes, cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy A_eff failed");
  cuda_check(cudaMemcpy(d_gamma_eff, gamma_eff.data(), cell_bytes,
                        cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy gamma_eff failed");
  cuda_check(cudaMemcpy(d_cell_is_void, state.cell_is_void.data(), mask_bytes,
                        cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy cell_is_void failed");
  cuda_check(cudaMemcpy(d_diffusion_cell, diffusion_cell.data(), mask_bytes,
                        cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy diffusion_cell failed");
  if (holo_updates_material_directly) {
    cuda_check(cudaMemcpy(d_holo_source_cell, state.holo_core_mask.data(), mask_bytes,
                          cudaMemcpyHostToDevice),
               "inject_radiation_source_terms copy holo_source_cell failed");
  }
  cuda_check(cudaMemset(d_reduction, 0, 2 * sizeof(double)),
             "inject_radiation_source_terms zero reduction failed");
  // Floor and skipped energies per cell ([0, n) and [n, 2n)), summed in a
  // fixed order into d_reduction below (2026-09-24; atomicAdd before).
  double* d_ledger_cells = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_ledger_cells",
      2 * static_cast<std::size_t>(n_cells) * sizeof(double)));
  cuda_check(cudaMemset(d_ledger_cells, 0, 2 * static_cast<std::size_t>(n_cells) * sizeof(double)),
             "inject_radiation_source_terms zero ledger cells failed");
  cuda_check(cudaMemset(d_clamp_count, 0, sizeof(int)),
             "inject_radiation_source_terms zero clamp_count failed");

  const materials::CellEOSTableSelector cell_tables =
      source_cell_table_selector(state, cfg, eos_ctx);
  const int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  inject_radiation_source_terms_kernel<<<blocks, threads>>>(
      state.ee.data(), state.Te.data(), state.Pe.data(), state.ei.data(),
      state.Ti.data(), state.Pi.data(), state.delta_E_rad_prev.data(),
      state.rho.data(), state.mass.data(), state.zbar.data(),
      has_table_cv_e ? state.cv_e.data() : nullptr, d_applied_delta_E, d_A_eff,
      d_gamma_eff, d_cell_is_void, d_diffusion_cell, d_holo_source_cell, n_cells, te_floor,
      ti_floor, use_two_temp, has_table_eos, use_first_cv_override,
      mat0.cv_e_override, mat0.eos_T_ref_eV, table_views.ion,
      table_views.electron, table_views.total, cell_tables, d_ledger_cells,
      d_ledger_cells + n_cells, d_clamp_count,
      (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative") ? 1 : 0);
  cuda_check(cudaGetLastError(),
             "inject_radiation_source_terms kernel launch failed");
  core::deterministic_sum(d_ledger_cells, n_cells, &d_reduction[0], false);
  core::deterministic_sum(d_ledger_cells + n_cells, n_cells, &d_reduction[1], false);
  if (verbose_subphase_timing) {
    cuda_check(core::debug_kernel_sync(),
               "inject_radiation_source_terms kernel synchronize failed");
  }
  mark_subphase(gpu_ms);

  double host_reduction[2] = {0.0, 0.0};
  int local_clamp_count = 0;
  cuda_check(cudaMemcpy(host_reduction, d_reduction, 2 * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "inject_radiation_source_terms copy reduction failed");
  cuda_check(cudaMemcpy(&local_clamp_count, d_clamp_count, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "inject_radiation_source_terms copy clamp_count failed");
  mark_subphase(reduction_ms);

  const double floor_energy = host_reduction[0];
  const double skipped_energy = host_reduction[1];
  accumulate_floor_and_clamp(E_floor_injected, clamp_count, floor_energy, local_clamp_count);
  mark_subphase(finalize_ms);
  log_subphase();
  return skipped_energy;
}

double inject_radiation_source_terms(core::State& state,
                                     const core::Config& cfg,
                                     const double dt,
                                     double* E_floor_injected,
                                     int* clamp_count,
                                     const std::vector<double>* sigma_R_max) {
  return inject_radiation_source_terms_impl(
      state, cfg, dt, E_floor_injected, clamp_count, sigma_R_max, nullptr);
}

double inject_radiation_source_terms(core::State& state,
                                     const core::Config& cfg,
                                     const double dt,
                                     double* E_floor_injected,
                                     int* clamp_count,
                                     const std::vector<double>* sigma_R_max,
                                     const hydro::HydroEOSContext* eos_ctx) {
  return inject_radiation_source_terms_impl(
      state, cfg, dt, E_floor_injected, clamp_count, sigma_R_max, eos_ctx);
}

}  // namespace tenryu::coupling
