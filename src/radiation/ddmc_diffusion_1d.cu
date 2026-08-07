#include "radiation/ddmc_diffusion_1d.cuh"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kFourPi = 12.56637061435917295385;
constexpr int kDDMCPicardIterations = 2;

struct DDMCBoundaryLeak1D {
  double left_sink = 0.0;
  double right_sink = 0.0;
  int left_neighbor = -1;
  int right_neighbor = -1;
  bool left_to_imc = false;
  bool right_to_imc = false;
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

double internal_face_coupling_1d(const DDMCCoefficients& coefficients,
                                 const std::vector<double>& node_r,
                                 const std::vector<double>& cell_width,
                                 const int face,
                                 const int group) {
  TENRYU_ASSERT(face > 0, "internal_face_coupling_1d requires face > 0");
  TENRYU_ASSERT(face < static_cast<int>(node_r.size()) - 1,
                "internal_face_coupling_1d requires interior face");
  const int c_left = face - 1;
  const int c_right = face;
  const double dx_left = std::max(cell_width[static_cast<std::size_t>(c_left)], 0.0);
  const double dx_right = std::max(cell_width[static_cast<std::size_t>(c_right)], 0.0);
  if (!(dx_left > 0.0) || !(dx_right > 0.0)) {
    return 0.0;
  }

  const double sigma_left = std::max(coefficients.get_face_sigma_minus(face, group), 1.0e-30);
  const double sigma_right = std::max(coefficients.get_face_sigma_plus(face, group), 1.0e-30);
  const double denom = 3.0 * (sigma_left * dx_left + sigma_right * dx_right);
  if (!(denom > 0.0)) {
    return 0.0;
  }

  const double r_face = std::max(node_r[static_cast<std::size_t>(face)], 0.0);
  const double area = kFourPi * r_face * r_face;
  return 2.0 * area * core::constants::c_light / denom;
}

double thermal_emission_source_1d(const PlanckTable& planck,
                                  const int group,
                                  const double sigma_a,
                                  const double T_eV,
                                  const double volume) {
  if (!(sigma_a > 0.0) || !(volume > 0.0)) {
    return 0.0;
  }

  const double T = std::max(T_eV, 0.0);
  const double T2 = T * T;
  const double T4 = T2 * T2;
  const double b_g = std::max(planck.interpolate_b_host(group, T), 0.0);
  return core::constants::c_light * sigma_a * core::constants::a_eV * T4 * b_g * volume;
}

void solve_tridiagonal(std::vector<double>& lower,
                       std::vector<double>& diag,
                       std::vector<double>& upper,
                       std::vector<double>& rhs,
                       std::vector<double>& sol) {
  const int n = static_cast<int>(diag.size());
  sol.assign(static_cast<std::size_t>(n), 0.0);
  if (n <= 0) {
    return;
  }

  for (int i = 1; i < n; ++i) {
    const double denom = diag[static_cast<std::size_t>(i - 1)];
    TENRYU_ASSERT(denom > 0.0, "DDMC implicit diffusion requires positive tridiagonal pivot");
    const double factor = lower[static_cast<std::size_t>(i)] / denom;
    diag[static_cast<std::size_t>(i)] -= factor * upper[static_cast<std::size_t>(i - 1)];
    rhs[static_cast<std::size_t>(i)] -= factor * rhs[static_cast<std::size_t>(i - 1)];
    lower[static_cast<std::size_t>(i)] = 0.0;
  }

  TENRYU_ASSERT(diag.back() > 0.0,
                "DDMC implicit diffusion requires positive tridiagonal pivot");
  sol.back() = rhs.back() / diag.back();
  for (int i = n - 2; i >= 0; --i) {
    TENRYU_ASSERT(diag[static_cast<std::size_t>(i)] > 0.0,
                  "DDMC implicit diffusion requires positive tridiagonal pivot");
    sol[static_cast<std::size_t>(i)] =
        (rhs[static_cast<std::size_t>(i)] -
         upper[static_cast<std::size_t>(i)] * sol[static_cast<std::size_t>(i + 1)]) /
        diag[static_cast<std::size_t>(i)];
  }
}

}  // namespace

DDMCDiffusionSolveResult solve_ddmc_diffusion_1d(
    const ModeSelector& mode_selector,
    const DDMCCoefficients& coefficients,
    const PlanckTable& planck,
    const std::vector<double>& node_r,
    const std::vector<double>& cell_vol,
    const std::vector<double>& sigma_a_eff,
    const std::vector<double>& Te,
    const std::vector<double>& cell_heat_capacity,
    const std::vector<double>& rad_E_prev,
    double* d_rad_dep,
    double* d_rad_emit,
    double* d_rad_E_tally,
    double* d_E_escape,
    const double dt) {
  DDMCDiffusionSolveResult out{};
  const int n_cells = static_cast<int>(mode_selector.n_cells());
  const int n_groups = mode_selector.n_groups();
  TENRYU_ASSERT(dt > 0.0, "solve_ddmc_diffusion_1d requires dt > 0");
  TENRYU_ASSERT(d_rad_dep != nullptr, "solve_ddmc_diffusion_1d requires d_rad_dep");
  TENRYU_ASSERT(d_rad_E_tally != nullptr, "solve_ddmc_diffusion_1d requires d_rad_E_tally");
  TENRYU_ASSERT(d_E_escape != nullptr, "solve_ddmc_diffusion_1d requires d_E_escape");
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == n_cells + 1,
                "solve_ddmc_diffusion_1d node_r size mismatch");
  TENRYU_ASSERT(static_cast<int>(cell_vol.size()) == n_cells,
                "solve_ddmc_diffusion_1d cell_vol size mismatch");
  TENRYU_ASSERT(static_cast<int>(Te.size()) == n_cells,
                "solve_ddmc_diffusion_1d Te size mismatch");
  TENRYU_ASSERT(static_cast<int>(cell_heat_capacity.size()) == n_cells,
                "solve_ddmc_diffusion_1d cell_heat_capacity size mismatch");
  TENRYU_ASSERT(static_cast<int>(sigma_a_eff.size()) == n_cells * n_groups,
                "solve_ddmc_diffusion_1d sigma_a_eff size mismatch");
  TENRYU_ASSERT(static_cast<int>(rad_E_prev.size()) == n_cells * n_groups,
                "solve_ddmc_diffusion_1d rad_E_prev size mismatch");
  TENRYU_ASSERT(d_rad_emit != nullptr, "solve_ddmc_diffusion_1d requires d_rad_emit");

  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  std::vector<double> host_rad_dep(n_cell_groups, 0.0);
  std::vector<double> host_rad_emit(n_cell_groups, 0.0);
  std::vector<double> host_rad_E_tally(n_cell_groups, 0.0);
  std::vector<double> host_E_escape(static_cast<std::size_t>(n_groups), 0.0);
  cuda_check(cudaMemcpy(host_rad_dep.data(),
                        d_rad_dep,
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyDeviceToHost),
             "solve_ddmc_diffusion_1d copy rad_dep failed");
  cuda_check(cudaMemcpy(host_rad_emit.data(),
                        d_rad_emit,
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyDeviceToHost),
             "solve_ddmc_diffusion_1d copy rad_emit failed");
  cuda_check(cudaMemcpy(host_rad_E_tally.data(),
                        d_rad_E_tally,
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyDeviceToHost),
             "solve_ddmc_diffusion_1d copy rad_E_tally failed");
  cuda_check(cudaMemcpy(host_E_escape.data(),
                        d_E_escape,
                        sizeof(double) * host_E_escape.size(),
                        cudaMemcpyDeviceToHost),
             "solve_ddmc_diffusion_1d copy E_escape failed");

  std::vector<double> cell_width(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    cell_width[static_cast<std::size_t>(c)] =
        std::max(node_r[static_cast<std::size_t>(c + 1)] -
                     node_r[static_cast<std::size_t>(c)],
                 0.0);
  }

  std::vector<double> picard_temperature(Te.begin(), Te.end());
  std::vector<double> delta_E_total(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<std::uint8_t> ddmc_cell_touched(static_cast<std::size_t>(n_cells), 0U);

  for (int iter = 0; iter < kDDMCPicardIterations; ++iter) {
    const bool final_iteration = (iter + 1 == kDDMCPicardIterations);
    std::fill(delta_E_total.begin(), delta_E_total.end(), 0.0);

    for (int g = 0; g < n_groups; ++g) {
      std::vector<int> ddmc_cells;
      ddmc_cells.reserve(static_cast<std::size_t>(n_cells));
      for (int c = 0; c < n_cells; ++c) {
        if (mode_selector.get_mode(c, g) == TransportMode::DDMC) {
          ddmc_cells.push_back(c);
        }
      }
      if (ddmc_cells.empty()) {
        continue;
      }

      if (iter == 0) {
        out.solved_cell_groups += static_cast<std::int64_t>(ddmc_cells.size());
      }

      const int n_ddmc = static_cast<int>(ddmc_cells.size());
      std::vector<double> lower(static_cast<std::size_t>(n_ddmc), 0.0);
      std::vector<double> diag(static_cast<std::size_t>(n_ddmc), 0.0);
      std::vector<double> upper(static_cast<std::size_t>(n_ddmc), 0.0);
      std::vector<double> rhs(static_cast<std::size_t>(n_ddmc), 0.0);
      std::vector<double> volume(static_cast<std::size_t>(n_ddmc), 0.0);
      std::vector<double> sigma_a(static_cast<std::size_t>(n_ddmc), 0.0);
      std::vector<double> energy_old(static_cast<std::size_t>(n_ddmc), 0.0);
      std::vector<double> emission_energy(static_cast<std::size_t>(n_ddmc), 0.0);
      std::vector<DDMCBoundaryLeak1D> boundary(static_cast<std::size_t>(n_ddmc));
      std::vector<double> solution;

      for (int i = 0; i < n_ddmc; ++i) {
        const int cell = ddmc_cells[static_cast<std::size_t>(i)];
        const std::size_t c_us = static_cast<std::size_t>(cell);
        const std::size_t key =
            c_us * static_cast<std::size_t>(n_groups) + static_cast<std::size_t>(g);
        const double V_i = std::max(cell_vol[c_us], 0.0);
        const double sigma_a_i = std::max(sigma_a_eff[key], 0.0);
        const double E_old = std::max(rad_E_prev[key], 0.0);
        const auto& cell_data = coefficients.get_cell_data(cell, g);

        ddmc_cell_touched[c_us] = 1U;
        volume[static_cast<std::size_t>(i)] = V_i;
        sigma_a[static_cast<std::size_t>(i)] = sigma_a_i;
        energy_old[static_cast<std::size_t>(i)] = E_old;

        if (!(V_i > 0.0)) {
          diag[static_cast<std::size_t>(i)] = 1.0;
          rhs[static_cast<std::size_t>(i)] = 0.0;
          continue;
        }

        diag[static_cast<std::size_t>(i)] =
            V_i / dt + core::constants::c_light * sigma_a_i * V_i;

        if (i > 0 && ddmc_cells[static_cast<std::size_t>(i - 1)] == cell - 1) {
          const double coupling = internal_face_coupling_1d(coefficients,
                                                            node_r,
                                                            cell_width,
                                                            cell,
                                                            g);
          if (coupling > 0.0) {
            lower[static_cast<std::size_t>(i)] = -coupling;
            diag[static_cast<std::size_t>(i)] += coupling;
          }
        } else if (cell_data.bc_left == DDMCBoundaryType::Vacuum ||
                   cell_data.bc_left == DDMCBoundaryType::Interface) {
          const double sink =
              core::constants::c_light * V_i * std::max(cell_data.sigma_leak_left, 0.0);
          boundary[static_cast<std::size_t>(i)].left_sink = sink;
          diag[static_cast<std::size_t>(i)] += sink;
          if (cell_data.bc_left == DDMCBoundaryType::Interface) {
            const int neighbor = cell_data.neighbor_face[0];
            if (neighbor >= 0 && neighbor < n_cells &&
                mode_selector.get_mode(neighbor, g) == TransportMode::IMC) {
              boundary[static_cast<std::size_t>(i)].left_neighbor = neighbor;
              boundary[static_cast<std::size_t>(i)].left_to_imc = true;
            }
          }
        }

        if (i + 1 < n_ddmc && ddmc_cells[static_cast<std::size_t>(i + 1)] == cell + 1) {
          const double coupling = internal_face_coupling_1d(coefficients,
                                                            node_r,
                                                            cell_width,
                                                            cell + 1,
                                                            g);
          if (coupling > 0.0) {
            upper[static_cast<std::size_t>(i)] = -coupling;
            diag[static_cast<std::size_t>(i)] += coupling;
          }
        } else if (cell_data.bc_right == DDMCBoundaryType::Vacuum ||
                   cell_data.bc_right == DDMCBoundaryType::Interface) {
          const double sink =
              core::constants::c_light * V_i * std::max(cell_data.sigma_leak_right, 0.0);
          boundary[static_cast<std::size_t>(i)].right_sink = sink;
          diag[static_cast<std::size_t>(i)] += sink;
          if (cell_data.bc_right == DDMCBoundaryType::Interface) {
            const int neighbor = cell_data.neighbor_face[1];
            if (neighbor >= 0 && neighbor < n_cells &&
                mode_selector.get_mode(neighbor, g) == TransportMode::IMC) {
              boundary[static_cast<std::size_t>(i)].right_neighbor = neighbor;
              boundary[static_cast<std::size_t>(i)].right_to_imc = true;
            }
          }
        }
      }

      for (int i = 0; i < n_ddmc; ++i) {
        const int cell = ddmc_cells[static_cast<std::size_t>(i)];
        const std::size_t c_us = static_cast<std::size_t>(cell);
        const double source =
            thermal_emission_source_1d(planck,
                                       g,
                                       sigma_a[static_cast<std::size_t>(i)],
                                       picard_temperature[c_us],
                                       volume[static_cast<std::size_t>(i)]);
        emission_energy[static_cast<std::size_t>(i)] = source * dt;
        rhs[static_cast<std::size_t>(i)] =
            volume[static_cast<std::size_t>(i)] * energy_old[static_cast<std::size_t>(i)] / dt +
            source;
      }

      solve_tridiagonal(lower, diag, upper, rhs, solution);

      for (int i = 0; i < n_ddmc; ++i) {
        const int cell = ddmc_cells[static_cast<std::size_t>(i)];
        const std::size_t c_us = static_cast<std::size_t>(cell);
        const std::size_t key =
            c_us * static_cast<std::size_t>(n_groups) + static_cast<std::size_t>(g);
        const double V_i = volume[static_cast<std::size_t>(i)];
        const double sigma_a_i = sigma_a[static_cast<std::size_t>(i)];
        const double E_new = std::max(solution[static_cast<std::size_t>(i)], 0.0);
        const double absorbed =
            core::constants::c_light * sigma_a_i * E_new * V_i * dt;
        delta_E_total[c_us] += absorbed - emission_energy[static_cast<std::size_t>(i)];

        if (!final_iteration) {
          continue;
        }

        host_rad_E_tally[key] = core::constants::c_light * E_new * V_i * dt;
        host_rad_dep[key] += absorbed;
        host_rad_emit[key] = emission_energy[static_cast<std::size_t>(i)];

        const double leak_left =
            std::max(boundary[static_cast<std::size_t>(i)].left_sink, 0.0) * E_new * dt;
        if (leak_left > 0.0 && boundary[static_cast<std::size_t>(i)].left_to_imc) {
          const int neighbor = boundary[static_cast<std::size_t>(i)].left_neighbor;
          const std::size_t neighbor_key =
              static_cast<std::size_t>(neighbor) * static_cast<std::size_t>(n_groups) +
              static_cast<std::size_t>(g);
          host_rad_dep[neighbor_key] += leak_left;
        } else {
          host_E_escape[static_cast<std::size_t>(g)] += leak_left;
          out.escaped_energy += leak_left;
        }

        const double leak_right =
            std::max(boundary[static_cast<std::size_t>(i)].right_sink, 0.0) * E_new * dt;
        if (leak_right > 0.0 && boundary[static_cast<std::size_t>(i)].right_to_imc) {
          const int neighbor = boundary[static_cast<std::size_t>(i)].right_neighbor;
          const std::size_t neighbor_key =
              static_cast<std::size_t>(neighbor) * static_cast<std::size_t>(n_groups) +
              static_cast<std::size_t>(g);
          host_rad_dep[neighbor_key] += leak_right;
        } else {
          host_E_escape[static_cast<std::size_t>(g)] += leak_right;
          out.escaped_energy += leak_right;
        }
      }
    }

    if (final_iteration) {
      continue;
    }

    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      picard_temperature[c_us] = std::max(Te[c_us], 0.0);
      if (ddmc_cell_touched[c_us] == 0U) {
        continue;
      }

      const double heat_capacity = std::max(cell_heat_capacity[c_us], 0.0);
      if (heat_capacity > 1.0e-30 && std::isfinite(delta_E_total[c_us])) {
        picard_temperature[c_us] += delta_E_total[c_us] / heat_capacity;
      }
      picard_temperature[c_us] = std::max(picard_temperature[c_us], 0.0);
    }
  }

  cuda_check(cudaMemcpy(d_rad_dep,
                        host_rad_dep.data(),
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyHostToDevice),
             "solve_ddmc_diffusion_1d upload rad_dep failed");
  cuda_check(cudaMemcpy(d_rad_emit,
                        host_rad_emit.data(),
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyHostToDevice),
             "solve_ddmc_diffusion_1d upload rad_emit failed");
  cuda_check(cudaMemcpy(d_rad_E_tally,
                        host_rad_E_tally.data(),
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyHostToDevice),
             "solve_ddmc_diffusion_1d upload rad_E_tally failed");
  cuda_check(cudaMemcpy(d_E_escape,
                        host_E_escape.data(),
                        sizeof(double) * host_E_escape.size(),
                        cudaMemcpyHostToDevice),
             "solve_ddmc_diffusion_1d upload E_escape failed");
  return out;
}

}  // namespace tenryu::radiation
