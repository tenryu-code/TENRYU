// Retired with the Monte Carlo radiation on 2026-09-29: the 1D S_N pieces of the HOLO S_N closure, cut out of the
// files below as they were at 5bc8f6ce3. Not built.
//   src/radiation/sn_transport_gpu.cu:  the 1D branch of solve_sn_material_coupling_gpu (IMEX source iteration with
//     DSA and the material Newton), solve_sn_transport_1d_gpu (the 1D GPU sweep, used only by a test against the CPU
//     sweep of sn_transport_1d.cpp) and the helpers only they used;
//   src/radiation/sn_transport_gpu.cuh: SNTransport1DGPUInputs and the matter-update fields of
//     SNMaterialCouplingGPUInputs (ee, rho, cv_e, sigma_R, Te_old, electron_eos, planck_table_cpu, dim,
//     update_material, cv_e_const, Cv_e_const);
//   src/radiation/sn_material_newton_gpu.cu: the SNMaterialCouplingGPUInputs overload of
//     solve_sn_material_temperature_newton_gpu.

// ---- sn_transport_gpu.cuh

struct SNTransport1DGPUInputs {
  const double* sigma_a = nullptr;          // [n_cells * n_groups], [1/cm]
  const double* sigma_s = nullptr;          // [n_cells * n_groups], [1/cm]
  const double* source_emission = nullptr;  // [n_cells * n_groups], [erg/cm^3/s]
  const double* node_r = nullptr;           // [n_cells + 1], [cm]
  const double* vol = nullptr;              // [n_cells], [cm^3]
  double* E_out = nullptr;                  // [n_cells * n_groups], [erg/cm^3]
  double* P_rr_out = nullptr;               // [n_cells * n_groups], [erg/cm^3]
  double* chi_out = nullptr;                // [n_cells * n_groups]
  double* psi_bar = nullptr;                // optional [n_groups * n_angles * n_cells]
  int n_cells = 0;
  int n_groups = 0;
  double dt = 0.0;
};


// ---- sn_transport_gpu.cu: removed helpers and solve_sn_transport_1d_gpu



__host__ __device__ inline double sn_mass_heat_capacity(
    const double rho,
    const double cv_e_value,
    const double cv_e_const,
    const double Cv_e_const) {
  const double cv_cell = finite_or_zero(cv_e_value);
  if (cv_cell > 0.0) {
    return cv_cell;
  }
  const double Cv_const = finite_or_zero(Cv_e_const);
  const double rho_c = nonnegative_finite(rho);
  if (Cv_const > 0.0 && rho_c > 0.0) {
    return Cv_const / rho_c;
  }
  return finite_or_zero(cv_e_const);
}

__host__ __device__ inline double sn_volume_heat_capacity(
    const double rho,
    const double cv_e_value,
    const double cv_e_const,
    const double Cv_e_const) {
  const double cv_cell = finite_or_zero(cv_e_value);
  const double rho_c = nonnegative_finite(rho);
  if (cv_cell > 0.0) {
    return rho_c * cv_cell;
  }
  const double Cv_const = finite_or_zero(Cv_e_const);
  if (Cv_const > 0.0) {
    return Cv_const;
  }
  return rho_c * finite_or_zero(cv_e_const);
}

__host__ __device__ inline std::size_t psi_index(const int group,
                                                 const int angle,
                                                 const int cell,
                                                 const int n_angles,
                                                 const int n_cells) {
  return (static_cast<std::size_t>(group) * static_cast<std::size_t>(n_angles) +
          static_cast<std::size_t>(angle)) *
             static_cast<std::size_t>(n_cells) +
         static_cast<std::size_t>(cell);
}

__host__ __device__ inline double face_area_1d(const double r) {
  const double rr = safe_radius(r);
  return kFourPi * rr * rr;
}

__host__ __device__ inline double shell_volume_1d(const double r_in,
                                                  const double r_out) {
  const double rin = safe_radius(r_in);
  const double rout = safe_radius(r_out);
  const double volume = kFourPiOverThree * (rout * rout * rout - rin * rin * rin);
  return fmax(volume, 0.0);
}

std::vector<double> angular_coefficients(const std::vector<double>& mu,
                                         const std::vector<double>& weight) {
  const int n_angles = static_cast<int>(mu.size());
  std::vector<double> alpha(static_cast<std::size_t>(n_angles + 1), 0.0);
  for (int n = 0; n < n_angles; ++n) {
    alpha[static_cast<std::size_t>(n + 1)] =
        alpha[static_cast<std::size_t>(n)] -
        mu[static_cast<std::size_t>(n)] * weight[static_cast<std::size_t>(n)];
  }
  alpha.front() = 0.0;
  alpha.back() = 0.0;
  for (double& value : alpha) {
    if (std::abs(value) < 1.0e-14) {
      value = 0.0;
    }
    value = std::max(value, 0.0);
  }
  return alpha;
}

__global__ void sn_update_material_energy_kernel(
    double* __restrict__ ee,
    const double* __restrict__ Te_new,
    const double* __restrict__ Te_old,
    const double* __restrict__ rho,
    const double* __restrict__ cv_e,
    const int n_cells,
    const double temperature_floor,
    const double cv_e_const,
    const double Cv_e_const) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double T_new = fmax(finite_or_zero(Te_new[c]), temperature_floor);
  const double T_old = fmax(finite_or_zero(Te_old[c]), temperature_floor);
  const double cv_mass = sn_mass_heat_capacity(
      rho[c], (cv_e != nullptr) ? cv_e[c] : 0.0, cv_e_const, Cv_e_const);
  const double delta_ee = cv_mass * (T_new - T_old);
  if (cv_mass > 0.0 && sn_finite(delta_ee)) {
    ee[c] += delta_ee;
  }
}

__global__ void sn_fill_kernel(double* __restrict__ values,
                               const int n_values,
                               const double value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_values) {
    values[idx] = value;
  }
}

double positive_harmonic_mean(const double a, const double b) {
  const double aa = std::max(finite_or_zero(a), kDsaSigmaFloor);
  const double bb = std::max(finite_or_zero(b), kDsaSigmaFloor);
  return 2.0 / (1.0 / aa + 1.0 / bb);
}

void solve_dsa_correction_1d(const double* phi_half,
                             const double* phi_old,
                             const double* sigma_a,
                             const double* sigma_s,
                             const double* node_r,
                             const double* vol,
                             double* delta_phi,
                             const int n_cells,
                             const int n_groups) {
  const std::size_t n_total =
      static_cast<std::size_t>(std::max(n_cells, 0)) *
      static_cast<std::size_t>(std::max(n_groups, 0));
  if (delta_phi == nullptr) {
    return;
  }
  std::fill(delta_phi, delta_phi + n_total, 0.0);
  if (phi_half == nullptr || phi_old == nullptr || sigma_a == nullptr ||
      sigma_s == nullptr || node_r == nullptr || vol == nullptr ||
      n_cells <= 0 || n_groups <= 0) {
    return;
  }

  std::vector<double> lower(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> diag(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> upper(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> rhs(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> diffusion(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> correction(static_cast<std::size_t>(n_cells), 0.0);

  for (int g = 0; g < n_groups; ++g) {
    std::fill(lower.begin(), lower.end(), 0.0);
    std::fill(diag.begin(), diag.end(), 0.0);
    std::fill(upper.begin(), upper.end(), 0.0);
    std::fill(rhs.begin(), rhs.end(), 0.0);
    std::fill(correction.begin(), correction.end(), 0.0);

    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const std::size_t cg = cell_group_index(c, g, n_groups);
      const double sigma_abs = std::max(finite_or_zero(sigma_a[cg]), 0.0);
      const double sigma_scat = std::max(finite_or_zero(sigma_s[cg]), 0.0);
      const double sigma_t = std::max(sigma_abs + sigma_scat, kDsaSigmaFloor);
      diffusion[c_us] = 1.0 / (3.0 * sigma_t);
      diag[c_us] = sigma_abs;
      rhs[c_us] =
          0.5 * sigma_scat *
          (finite_or_zero(phi_half[cg]) - finite_or_zero(phi_old[cg]));
      if (!sn_finite(rhs[c_us])) {
        rhs[c_us] = 0.0;
      }
    }

    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double V_input = std::max(finite_or_zero(vol[c_us]), 0.0);
      const double V = (V_input > 0.0)
                           ? V_input
                           : shell_volume_1d(node_r[c], node_r[c + 1]);
      if (!(V > 0.0)) {
        lower[c_us] = 0.0;
        diag[c_us] = 1.0;
        upper[c_us] = 0.0;
        rhs[c_us] = 0.0;
        continue;
      }

      const double center =
          0.5 * (safe_radius(node_r[c]) + safe_radius(node_r[c + 1]));
      if (c + 1 < n_cells) {
        const double center_right =
            0.5 * (safe_radius(node_r[c + 1]) + safe_radius(node_r[c + 2]));
        const double dr = center_right - center;
        const double A = face_area_1d(node_r[c + 1]);
        const double d_face =
            positive_harmonic_mean(diffusion[c_us], diffusion[c_us + 1U]);
        const double coeff = (dr > 0.0) ? (A * d_face / (dr * V)) : 0.0;
        if (sn_finite(coeff) && coeff > 0.0) {
          diag[c_us] += coeff;
          upper[c_us] -= coeff;
        }
      }
      if (c > 0) {
        const double center_left =
            0.5 * (safe_radius(node_r[c - 1]) + safe_radius(node_r[c]));
        const double dr = center - center_left;
        const double A = face_area_1d(node_r[c]);
        const double d_face =
            positive_harmonic_mean(diffusion[c_us - 1U], diffusion[c_us]);
        const double coeff = (dr > 0.0) ? (A * d_face / (dr * V)) : 0.0;
        if (sn_finite(coeff) && coeff > 0.0) {
          diag[c_us] += coeff;
          lower[c_us] -= coeff;
        }
      }
    }

    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      if (!sn_finite(lower[c_us])) {
        lower[c_us] = 0.0;
      }
      if (!sn_finite(upper[c_us])) {
        upper[c_us] = 0.0;
      }
      if (!sn_finite(diag[c_us]) || diag[c_us] <= 0.0) {
        lower[c_us] = 0.0;
        diag[c_us] = 1.0;
        upper[c_us] = 0.0;
        rhs[c_us] = 0.0;
      }
      if (!sn_finite(rhs[c_us])) {
        rhs[c_us] = 0.0;
      }
    }

    for (int c = 1; c < n_cells; ++c) {
      const std::size_t i = static_cast<std::size_t>(c);
      const std::size_t im1 = i - 1U;
      const double pivot =
          std::copysign(std::max(std::abs(diag[im1]), kDsaPivotFloor), diag[im1]);
      const double m = lower[i] / pivot;
      diag[i] -= m * upper[im1];
      rhs[i] -= m * rhs[im1];
    }

    for (int c = n_cells - 1; c >= 0; --c) {
      const std::size_t i = static_cast<std::size_t>(c);
      const double pivot =
          std::copysign(std::max(std::abs(diag[i]), kDsaPivotFloor), diag[i]);
      const double next_term =
          (c + 1 < n_cells) ? upper[i] * correction[i + 1U] : 0.0;
      correction[i] = (rhs[i] - next_term) / pivot;
      if (!sn_finite(correction[i])) {
        correction[i] = 0.0;
      }
    }

    for (int c = 0; c < n_cells; ++c) {
      delta_phi[cell_group_index(c, g, n_groups)] =
          correction[static_cast<std::size_t>(c)];
    }
  }
}

__global__ void sn_scalar_flux_to_E_kernel(const double* __restrict__ scalar_flux,
                                           double* __restrict__ E_out,
                                           const int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  E_out[idx] =
      fmax(finite_or_zero(scalar_flux[idx]), 0.0) /
      tenryu::core::constants::c_light;
}

__global__ void sn_sweep_1d_kernel(const double* __restrict__ sigma_a,
                                   const double* __restrict__ sigma_s,
                                   const double* __restrict__ source_emission,
                                   const double* __restrict__ scalar_flux,
                                   const double* __restrict__ node_r,
                                   const double* __restrict__ vol,
                                   const double* __restrict__ mu,
                                   const double* __restrict__ weights,
                                   const double* __restrict__ alpha_half,
                                   double* __restrict__ psi_bar,
                                   double* __restrict__ new_scalar_flux,
                                   double* __restrict__ E_out,
                                   double* __restrict__ P_rr_out,
                                   const int n_cells,
                                   const int n_groups,
                                   const int n_angles,
                                   const double dt) {
  const int g = blockIdx.x;
  if (g >= n_groups || threadIdx.x != 0) {
    return;
  }
  const double inv_cdt =
      (dt > 0.0) ? (1.0 / (tenryu::core::constants::c_light * dt)) : 0.0;
  extern __shared__ double shared[];
  double* angular_edge = shared;
  double* inner_boundary = shared + n_cells;

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cg = cell_group_index(c, g, n_groups);
    new_scalar_flux[cg] = 0.0;
    E_out[cg] = 0.0;
    P_rr_out[cg] = 0.0;
    angular_edge[c] = 0.0;
  }
  for (int n = 0; n < n_angles; ++n) {
    inner_boundary[n] = 0.0;
  }

  for (int n = 0; n < n_angles; ++n) {
    const double mu_n = mu[n];
    if (mu_n == 0.0) {
      continue;
    }
    const double abs_mu = fabs(mu_n);
    const double alpha_prev = alpha_half[n];
    const double alpha_next = alpha_half[n + 1];
    const double w = weights[n];
    const double mu2 = mu_n * mu_n;

    if (mu_n < 0.0) {
      double incoming = 0.0;
      for (int c = n_cells - 1; c >= 0; --c) {
        const std::size_t cg = cell_group_index(c, g, n_groups);
        const double area_in = face_area_1d(node_r[c]);
        const double area_out = face_area_1d(node_r[c + 1]);
        const double V_input = nonnegative_finite(vol[c]);
        const double V =
            (V_input > 0.0) ? V_input : shell_volume_1d(node_r[c], node_r[c + 1]);
        const double sigma_t =
            fmax(nonnegative_finite(sigma_a[cg]) + nonnegative_finite(sigma_s[cg]),
                 kSigmaFloor);
        const double sigma_eff = sigma_t + inv_cdt;
        const double source =
            0.5 * nonnegative_finite(source_emission[cg]) +
            0.5 * nonnegative_finite(sigma_s[cg]) *
                fmax(finite_or_zero(scalar_flux[cg]), 0.0);
        const double edge_in = fmax(angular_edge[c], 0.0);
        const double denom =
            2.0 * abs_mu * area_in + 2.0 * alpha_next + sigma_eff * V;
        double average = 0.0;
        if (denom > 0.0 && sn_finite(denom)) {
          const double numer =
              V * source + abs_mu * (area_in + area_out) * incoming +
              (alpha_prev + alpha_next) * edge_in;
          average = numer / denom;
        }
        average = fmax(finite_or_zero(average), 0.0);
        double outgoing = 2.0 * average - incoming;
        double edge_out = 2.0 * average - edge_in;
        if (!sn_finite(outgoing) || outgoing < 0.0) {
          outgoing = 0.0;
          average = 0.5 * incoming;  // conservative half-range fixup
        }
        if (!sn_finite(edge_out) || edge_out < 0.0) {
          edge_out = 0.0;
        }
        if (psi_bar != nullptr) {
          psi_bar[psi_index(g, n, c, n_angles, n_cells)] = average;
        }
        new_scalar_flux[cg] += w * average;
        P_rr_out[cg] += w * mu2 * average / tenryu::core::constants::c_light;
        angular_edge[c] = edge_out;
        incoming = outgoing;
      }
      inner_boundary[n] = incoming;
    } else {
      const int reflected = n_angles - 1 - n;
      double incoming =
          (reflected >= 0 && reflected < n_angles) ? fmax(inner_boundary[reflected], 0.0)
                                                   : 0.0;
      for (int c = 0; c < n_cells; ++c) {
        const std::size_t cg = cell_group_index(c, g, n_groups);
        const double area_in = face_area_1d(node_r[c]);
        const double area_out = face_area_1d(node_r[c + 1]);
        const double V_input = nonnegative_finite(vol[c]);
        const double V =
            (V_input > 0.0) ? V_input : shell_volume_1d(node_r[c], node_r[c + 1]);
        const double sigma_t =
            fmax(nonnegative_finite(sigma_a[cg]) + nonnegative_finite(sigma_s[cg]),
                 kSigmaFloor);
        const double sigma_eff = sigma_t + inv_cdt;
        const double source =
            0.5 * nonnegative_finite(source_emission[cg]) +
            0.5 * nonnegative_finite(sigma_s[cg]) *
                fmax(finite_or_zero(scalar_flux[cg]), 0.0);
        const double edge_in = fmax(angular_edge[c], 0.0);
        const double denom =
            2.0 * mu_n * area_out + 2.0 * alpha_next + sigma_eff * V;
        double average = 0.0;
        if (denom > 0.0 && sn_finite(denom)) {
          const double numer =
              V * source + mu_n * (area_in + area_out) * incoming +
              (alpha_prev + alpha_next) * edge_in;
          average = numer / denom;
        }
        average = fmax(finite_or_zero(average), 0.0);
        double outgoing = 2.0 * average - incoming;
        double edge_out = 2.0 * average - edge_in;
        if (!sn_finite(outgoing) || outgoing < 0.0) {
          outgoing = 0.0;
          average = 0.5 * incoming;  // conservative half-range fixup
        }
        if (!sn_finite(edge_out) || edge_out < 0.0) {
          edge_out = 0.0;
        }
        if (psi_bar != nullptr) {
          psi_bar[psi_index(g, n, c, n_angles, n_cells)] = average;
        }
        new_scalar_flux[cg] += w * average;
        P_rr_out[cg] += w * mu2 * average / tenryu::core::constants::c_light;
        angular_edge[c] = edge_out;
        incoming = outgoing;
      }
    }
  }

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cg = cell_group_index(c, g, n_groups);
    E_out[cg] =
        fmax(finite_or_zero(new_scalar_flux[cg]), 0.0) /
        tenryu::core::constants::c_light;
  }
}

void update_E_from_scalar_flux(const double* scalar_flux,
                               double* E_out,
                               const int n_total) {
  if (n_total <= 0) {
    return;
  }
  const int blocks = (n_total + kReduceBlock - 1) / kReduceBlock;
  sn_scalar_flux_to_E_kernel<<<blocks, kReduceBlock>>>(scalar_flux, E_out, n_total);
  cuda_check(cudaGetLastError(), "SN GPU scalar flux E update launch failed");
}

#ifdef TENRYU_DEBUG_CPU_FALLBACK
[[deprecated("Debug fallback only; production S_N material coupling uses GPU Newton")]]
void solve_material_temperature_newton_cpu(
    const SNMaterialCouplingGPUInputs& in,
    const int n_cells,
    const int n_groups,
    const double temperature_floor) {
  TENRYU_ASSERT(in.rho != nullptr, "SN IMEX Newton requires rho");
  TENRYU_ASSERT(in.ee != nullptr, "SN IMEX Newton requires ee");
  TENRYU_ASSERT(in.Te_old != nullptr, "SN IMEX Newton requires Te_old");
  TENRYU_ASSERT(in.planck_table_cpu != nullptr,
                "SN IMEX Newton requires CPU Planck table");
  TENRYU_ASSERT(in.planck_table_cpu->n_groups() == n_groups,
                "SN IMEX Newton Planck group count mismatch");

  const std::size_t n_cells_us = static_cast<std::size_t>(n_cells);
  const std::size_t n_total_us = n_cells_us * static_cast<std::size_t>(n_groups);
  const std::size_t cell_bytes = sizeof(double) * n_cells_us;
  const std::size_t total_bytes = sizeof(double) * n_total_us;

  std::vector<double> host_E(n_total_us, 0.0);
  std::vector<double> host_T(n_cells_us, 0.0);
  std::vector<double> host_T_old(n_cells_us, 0.0);
  std::vector<double> host_sigma_a(n_total_us, 0.0);
  std::vector<double> host_rho(n_cells_us, 0.0);
  std::vector<double> host_cv_e;
  if (in.cv_e != nullptr) {
    host_cv_e.resize(n_cells_us, 0.0);
  }

  cuda_check(cudaMemcpy(host_E.data(), in.E_out, total_bytes, cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy E failed");
  cuda_check(cudaMemcpy(host_T.data(), in.Te, cell_bytes, cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy Te failed");
  cuda_check(cudaMemcpy(host_T_old.data(),
                        in.Te_old,
                        cell_bytes,
                        cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy Te_old failed");
  cuda_check(cudaMemcpy(host_sigma_a.data(),
                        in.sigma_a,
                        total_bytes,
                        cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy sigma_a failed");
  cuda_check(cudaMemcpy(host_rho.data(), in.rho, cell_bytes, cudaMemcpyDeviceToHost),
             "SN IMEX Newton copy rho failed");
  if (!host_cv_e.empty()) {
    cuda_check(cudaMemcpy(host_cv_e.data(),
                          in.cv_e,
                          cell_bytes,
                          cudaMemcpyDeviceToHost),
               "SN IMEX Newton copy cv_e failed");
  }

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    host_T_old[c_us] =
        std::max(finite_or_zero(host_T_old[c_us]), temperature_floor);
    host_T[c_us] = host_T_old[c_us];
  }

  const double cv_e_const = finite_or_zero(in.cv_e_const);
  const double Cv_e_const = finite_or_zero(in.Cv_e_const);
  const double dt = in.dt;
  for (int iter = 0; iter < kSnTemperatureNewtonMaxIterations; ++iter) {
    double max_relative_update = 0.0;
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double rho_c = nonnegative_finite(host_rho[c_us]);
      const double cv_mass =
          (!host_cv_e.empty()) ? finite_or_zero(host_cv_e[c_us]) : 0.0;
      const double Cv =
          sn_volume_heat_capacity(rho_c, cv_mass, cv_e_const, Cv_e_const);
      if (!(Cv > 0.0)) {
        continue;
      }

      const double T_old = host_T_old[c_us];
      const double T = std::max(finite_or_zero(host_T[c_us]), temperature_floor);
      double F = Cv * (T - T_old) / dt;
      double dF = Cv / dt;
      const int base = c * n_groups;
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t cg = static_cast<std::size_t>(base + g);
        const double sigma = nonnegative_finite(host_sigma_a[cg]);
        if (!(sigma > 0.0)) {
          continue;
        }
        const double E = nonnegative_finite(host_E[cg]);
        const double c_sigma = tenryu::core::constants::c_light * sigma;
        const double b_g =
            std::max(in.planck_table_cpu->interpolate_b_host(g, T), 0.0);
        const double T4 = safe_temperature_pow4(T);
        const double B = tenryu::core::constants::a_eV * T4 * b_g;
        F -= c_sigma * (E - B);
        if (T > 0.0) {
          const double dBdT =
              4.0 * tenryu::core::constants::a_eV * (T4 / T) * b_g;
          dF += c_sigma * dBdT;
        }
      }

      if (!(sn_finite(F) && sn_finite(dF)) || !(dF > 0.0)) {
        continue;
      }
      const double dT = -F / dF;
      double T_next = T + dT;
      if (!sn_finite(T_next)) {
        T_next = temperature_floor;
      }
      T_next = std::max(T_next, temperature_floor);
      const double relative_update =
          std::abs(T_next - T) / std::max(std::abs(T_next), temperature_floor);
      max_relative_update = std::max(max_relative_update, relative_update);
      host_T[c_us] = T_next;
    }
    if (max_relative_update < kSnTemperatureNewtonTol) {
      break;
    }
  }

  cuda_check(cudaMemcpy(in.Te, host_T.data(), cell_bytes, cudaMemcpyHostToDevice),
             "SN IMEX Newton copy updated Te failed");
  const int blocks = (n_cells + kReduceBlock - 1) / kReduceBlock;
  sn_update_material_energy_kernel<<<blocks, kReduceBlock>>>(
      in.ee,
      in.Te,
      in.Te_old,
      in.rho,
      in.cv_e,
      n_cells,
      temperature_floor,
      cv_e_const,
      Cv_e_const);
  cuda_check(cudaGetLastError(), "SN IMEX Newton material energy update launch failed");
}  // namespace

SNTransportGPUResult solve_sn_transport_1d_gpu(
    const SNTransport1DGPUInputs& in,
    const SNTransportGPUConfig& config) {
  TENRYU_ASSERT(in.sigma_a != nullptr, "SN GPU 1D requires sigma_a");
  TENRYU_ASSERT(in.sigma_s != nullptr, "SN GPU 1D requires sigma_s");
  TENRYU_ASSERT(in.source_emission != nullptr, "SN GPU 1D requires source_emission");
  TENRYU_ASSERT(in.node_r != nullptr, "SN GPU 1D requires node_r");
  TENRYU_ASSERT(in.vol != nullptr, "SN GPU 1D requires vol");
  TENRYU_ASSERT(in.E_out != nullptr, "SN GPU 1D requires E_out");
  TENRYU_ASSERT(in.P_rr_out != nullptr, "SN GPU 1D requires P_rr_out");
  TENRYU_ASSERT(in.chi_out != nullptr, "SN GPU 1D requires chi_out");
  TENRYU_ASSERT(in.n_cells >= 0, "SN GPU 1D requires n_cells >= 0");
  TENRYU_ASSERT(in.n_groups >= 0, "SN GPU 1D requires n_groups >= 0");
  TENRYU_ASSERT(config.n_angles > 0 && (config.n_angles % 2) == 0,
                "SN GPU 1D n_angles must be positive and even");
  TENRYU_ASSERT(config.max_iterations >= 0,
                "SN GPU 1D max_iterations must be >= 0");
  TENRYU_ASSERT(config.convergence_tol >= 0.0,
                "SN GPU 1D convergence_tol must be >= 0");

  SNTransportGPUResult result{};
  result.n_directions = config.n_angles;
  const int n_total = in.n_cells * in.n_groups;
  if (in.n_cells == 0 || in.n_groups == 0) {
    result.converged = true;
    return result;
  }

  std::vector<double> mu;
  std::vector<double> weight;
  compute_gauss_legendre(config.n_angles, mu, weight);
  const std::vector<double> alpha = angular_coefficients(mu, weight);

  parallel::DeviceArray d_mu;
  parallel::DeviceArray d_weight;
  parallel::DeviceArray d_alpha;
  parallel::DeviceArray d_phi_a;
  parallel::DeviceArray d_phi_b;
  parallel::DeviceArray d_reduce;
  upload_vector(d_mu, mu, "SN GPU 1D copy mu failed");
  upload_vector(d_weight, weight, "SN GPU 1D copy weights failed");
  upload_vector(d_alpha, alpha, "SN GPU 1D copy alpha failed");

  const std::size_t phi_bytes = sizeof(double) * static_cast<std::size_t>(n_total);
  d_phi_a.resize(phi_bytes);
  d_phi_b.resize(phi_bytes);
  cuda_check(cudaMemset(d_phi_a.ptr, 0, phi_bytes), "SN GPU 1D zero phi failed");
  cuda_check(cudaMemset(d_phi_b.ptr, 0, phi_bytes), "SN GPU 1D zero phi_next failed");

  double* phi_old = d_phi_a.as<double>();
  double* phi_new = d_phi_b.as<double>();
  const int max_iterations = std::max(config.max_iterations, 1);
  const std::size_t shared_bytes =
      (static_cast<std::size_t>(in.n_cells) +
       static_cast<std::size_t>(config.n_angles)) *
      sizeof(double);
  for (int iter = 0; iter < max_iterations; ++iter) {
    sn_sweep_1d_kernel<<<in.n_groups, 1, shared_bytes>>>(
        in.sigma_a,
        in.sigma_s,
        in.source_emission,
        phi_old,
        in.node_r,
        in.vol,
        d_mu.as<double>(),
        d_weight.as<double>(),
        d_alpha.as<double>(),
        in.psi_bar,
        phi_new,
        in.E_out,
        in.P_rr_out,
        in.n_cells,
        in.n_groups,
        config.n_angles,
        0.0);
    cuda_check(cudaGetLastError(), "SN GPU 1D sweep launch failed");

    const double max_error =
        convergence_error(phi_new, phi_old, n_total, d_reduce);
    result.iterations = iter + 1;
    result.convergence_error = max_error;
    result.converged = result.convergence_error <= config.convergence_tol;
    if (result.converged) {
      break;
    }
    std::swap(phi_old, phi_new);
  }

  finalize_chi(in.E_out, in.P_rr_out, in.chi_out, n_total);
  return result;
}

// ---- sn_transport_gpu.cu: the 1D branch of solve_sn_material_coupling_gpu

  if (in.dim == 1) {
    TENRYU_ASSERT(config.n_angles > 0 && (config.n_angles % 2) == 0,
                  "SN IMEX DSA n_angles must be positive and even");

    result.n_directions = config.n_angles;
    std::vector<double> mu;
    std::vector<double> weight;
    compute_gauss_legendre(config.n_angles, mu, weight);
    const std::vector<double> alpha = angular_coefficients(mu, weight);

    parallel::DeviceArray d_mu;
    parallel::DeviceArray d_weight;
    parallel::DeviceArray d_alpha;
    parallel::DeviceArray d_phi_a;
    parallel::DeviceArray d_phi_b;
    parallel::DeviceArray d_reduce;
    upload_vector(d_mu, mu, "SN IMEX DSA copy mu failed");
    upload_vector(d_weight, weight, "SN IMEX DSA copy weights failed");
    upload_vector(d_alpha, alpha, "SN IMEX DSA copy alpha failed");

    const std::size_t phi_bytes =
        sizeof(double) * static_cast<std::size_t>(n_total);
    d_phi_a.resize(phi_bytes);
    d_phi_b.resize(phi_bytes);
    cuda_check(cudaMemset(d_phi_a.ptr, 0, phi_bytes), "SN IMEX DSA zero phi failed");
    cuda_check(cudaMemset(d_phi_b.ptr, 0, phi_bytes),
               "SN IMEX DSA zero phi_next failed");

    const std::size_t cell_bytes =
        sizeof(double) * static_cast<std::size_t>(n_cells);
    std::vector<double> host_sigma_a(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_sigma_s(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_phi_half(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_phi_old(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_delta_phi(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_phi_corrected(static_cast<std::size_t>(n_total), 0.0);
    std::vector<double> host_vol(static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> host_node_r(static_cast<std::size_t>(n_cells + 1), 0.0);

    cuda_check(cudaMemcpy(host_sigma_a.data(),
                          in.sigma_a,
                          sizeof(double) * host_sigma_a.size(),
                          cudaMemcpyDeviceToHost),
               "SN IMEX DSA copy sigma_a failed");
    cuda_check(cudaMemcpy(host_sigma_s.data(),
                          in.sigma_s,
                          sizeof(double) * host_sigma_s.size(),
                          cudaMemcpyDeviceToHost),
               "SN IMEX DSA copy sigma_s failed");
    cuda_check(cudaMemcpy(host_vol.data(),
                          in.vol,
                          cell_bytes,
                          cudaMemcpyDeviceToHost),
               "SN IMEX DSA copy vol failed");
    cuda_check(cudaMemcpy(host_node_r.data(),
                          in.node_r,
                          sizeof(double) * host_node_r.size(),
                          cudaMemcpyDeviceToHost),
               "SN IMEX DSA copy node_r failed");

    double* phi_old = d_phi_a.as<double>();
    double* phi_new = d_phi_b.as<double>();
    const double temperature_floor = std::max(config.temperature_floor_eV, 1.0e-12);
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(in.nr) +
         static_cast<std::size_t>(config.n_angles)) *
        sizeof(double);

    sn_build_source_kernel<<<blocks, kReduceBlock>>>(
        in.sigma_a,
        in.Te,
        d_source_emission.as<double>(),
        n_cells,
        in.n_groups,
        temperature_floor,
        in.planck);
    cuda_check(cudaGetLastError(), "SN IMEX DSA source build launch failed");

    const int max_iterations = 500;
    for (int iter = 0; iter < max_iterations; ++iter) {
      sn_sweep_1d_kernel<<<in.n_groups, 1, shared_bytes>>>(
          in.sigma_a,
          in.sigma_s,
          d_source_emission.as<double>(),
          phi_old,
          in.node_r,
          in.vol,
          d_mu.as<double>(),
          d_weight.as<double>(),
          d_alpha.as<double>(),
          nullptr,
          phi_new,
          in.E_out,
          in.P_rr_out,
          in.nr,
          in.n_groups,
          config.n_angles,
          in.dt);
      cuda_check(cudaGetLastError(), "SN IMEX source iteration sweep launch failed");

      const double max_error =
          convergence_error(phi_new, phi_old, n_total, d_reduce);
      result.iterations = iter + 1;
      result.convergence_error = max_error;
      result.converged = result.convergence_error <= config.convergence_tol;
      if (result.converged) {
        break;
      }
      std::swap(phi_old, phi_new);
    }

    if (in.update_material) {
      solve_sn_material_temperature_newton_gpu(in,
                                               n_cells,
                                               in.n_groups,
                                               temperature_floor);
    }
    const std::size_t total_bytes =
        sizeof(double) * static_cast<std::size_t>(n_total);
    cuda_check(cudaMemset(in.rad_dep, 0, total_bytes),
               "SN IMEX zero rad_dep after Newton failed");
    cuda_check(cudaMemset(in.rad_emit, 0, total_bytes),
               "SN IMEX zero rad_emit after Newton failed");
    if (in.coverage != nullptr) {
      sn_fill_kernel<<<blocks, kReduceBlock>>>(in.coverage, n_total, 1.0);
      cuda_check(cudaGetLastError(), "SN IMEX coverage fill launch failed");
    }

    finalize_chi(in.E_out, in.P_rr_out, in.chi_out, n_total);


// ---- sn_material_newton_gpu.cu

void solve_sn_material_temperature_newton_gpu(
    const SNMaterialCouplingGPUInputs& in,
    const int n_cells,
    const int n_groups,
    const double temperature_floor_eV) {
  SnMaterialNewton1DInputs wrapped{};
  wrapped.sigma_a = in.sigma_a;
  wrapped.rad_E = in.E_out;
  wrapped.rad_E_out = in.E_out;
  wrapped.rho = in.rho;
  wrapped.cv_e = in.cv_e;
  wrapped.vol = in.vol;
  wrapped.Te_old = in.Te_old;
  wrapped.Te = in.Te;
  wrapped.ee = in.ee;
  wrapped.rad_dep = in.rad_dep;
  wrapped.rad_emit = in.rad_emit;
  wrapped.planck = in.planck;
  wrapped.electron_eos = in.electron_eos;
  wrapped.n_cells = n_cells;
  wrapped.n_groups = n_groups;
  wrapped.dt = in.dt;
  wrapped.cv_e_const = in.cv_e_const;
  wrapped.Cv_e_const = in.Cv_e_const;
  wrapped.temperature_floor_eV = temperature_floor_eV;
  wrapped.c_begin = in.mpi_c_begin;
  wrapped.c_end = in.mpi_c_end;
  solve_sn_material_temperature_newton_gpu(wrapped);
}
